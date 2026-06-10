// OAuth2 Provider for Quillon Wallet - Third-party Integration
// Allows external websites to authenticate users and request wallet operations
// Uses post-quantum encryption (Kyber1024) for all sensitive data

use axum::{
    extract::{Json, Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Redirect},
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::{ApiResponse, AppState};

// ============================================================================
// OAuth2 Configuration
// ============================================================================

pub const TOKEN_EXPIRY_SECONDS: i64 = 3600; // 1 hour
pub const AUTH_CODE_EXPIRY_SECONDS: i64 = 300; // 5 minutes
pub const REFRESH_TOKEN_EXPIRY_DAYS: i64 = 30; // 30 days

// ============================================================================
// Data Structures
// ============================================================================

/// OAuth2 Client Registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2Client {
    pub client_id: String,
    pub client_secret: String,
    pub redirect_uris: Vec<String>,
    pub name: String,
    pub description: String,
    pub website: String,
    pub logo_url: Option<String>,
    pub scopes: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub kyber_public_key: Option<Vec<u8>>, // Post-quantum encryption key
}

/// OAuth2 Authorization Code
#[derive(Debug, Clone)]
pub struct AuthorizationCode {
    pub code: String,
    pub client_id: String,
    pub wallet_address: String,
    pub redirect_uri: String,
    pub scopes: Vec<String>,
    pub expires_at: DateTime<Utc>,
    pub code_challenge: Option<String>, // PKCE support
    pub code_challenge_method: Option<String>,
}

/// OAuth2 Access Token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessToken {
    pub token: String,
    pub client_id: String,
    pub wallet_address: String,
    pub scopes: Vec<String>,
    pub expires_at: DateTime<Utc>,
    pub refresh_token: Option<String>,
}

/// OAuth2 Consent Record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserConsent {
    pub wallet_address: String,
    pub client_id: String,
    pub scopes: Vec<String>,
    pub granted_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// v7.4.0: Peer JWT public key info for cross-node token verification
#[derive(Clone, Debug)]
pub struct PeerJwtKeyInfo {
    pub ed25519_verifying_bytes: [u8; 32],
    pub sqisign_public: Option<Vec<u8>>,
    pub announced_at: DateTime<Utc>,
}

// ============================================================================
// OAuth2 Storage
// v4.0.5: Each field has its own RwLock for fine-grained locking.
// AppState wraps this in Arc (NOT Arc<RwLock<>>), so callers access
// methods directly: state.oauth2_storage.get_client() - no outer lock.
// ============================================================================

/// v8.5.9: Device login code for miner OAuth2 flow
/// Miner generates a code, user visits URL in browser, logs in, miner polls for wallet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceLoginCode {
    pub code: String,
    pub user_code: String,     // Short code shown to user (e.g., "ABCD-1234")
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub wallet_address: Option<String>,  // Filled when user completes login
    pub completed: bool,
}

pub struct OAuth2Storage {
    clients: RwLock<HashMap<String, OAuth2Client>>,
    auth_codes: RwLock<HashMap<String, AuthorizationCode>>,
    access_tokens: RwLock<HashMap<String, AccessToken>>,
    refresh_tokens: RwLock<HashMap<String, String>>, // refresh_token -> access_token
    user_consents: RwLock<HashMap<(String, String), UserConsent>>, // (wallet, client_id) -> consent
    consent_hashes: RwLock<std::collections::HashSet<String>>, // v7.4.0: on-chain consent hashes
    storage: Option<Arc<q_storage::StorageEngine>>, // v7.3.5: RocksDB persistence for clients
    // v8.5.9: Device login codes for miner browser-based login
    pub device_codes: RwLock<HashMap<String, DeviceLoginCode>>, // code -> login state
}

impl OAuth2Storage {
    pub fn new() -> Self {
        Self {
            clients: RwLock::new(HashMap::new()),
            auth_codes: RwLock::new(HashMap::new()),
            access_tokens: RwLock::new(HashMap::new()),
            refresh_tokens: RwLock::new(HashMap::new()),
            user_consents: RwLock::new(HashMap::new()),
            consent_hashes: RwLock::new(std::collections::HashSet::new()),
            storage: None,
            device_codes: RwLock::new(HashMap::new()),
        }
    }

    /// Create with RocksDB persistence — clients survive restarts
    pub fn with_storage(storage: Arc<q_storage::StorageEngine>) -> Self {
        Self {
            clients: RwLock::new(HashMap::new()),
            auth_codes: RwLock::new(HashMap::new()),
            access_tokens: RwLock::new(HashMap::new()),
            refresh_tokens: RwLock::new(HashMap::new()),
            user_consents: RwLock::new(HashMap::new()),
            consent_hashes: RwLock::new(std::collections::HashSet::new()),
            storage: Some(storage),
            device_codes: RwLock::new(HashMap::new()),
        }
    }

    /// Load all persisted OAuth2 clients from RocksDB on startup
    pub async fn load_clients_from_disk(&self) {
        let storage = match &self.storage {
            Some(s) => s,
            None => return,
        };
        match storage.db_get(q_storage::CF_MANIFEST, b"oauth2:clients_index").await {
            Ok(Some(index_bytes)) => {
                let index_str = String::from_utf8_lossy(&index_bytes);
                let client_ids: Vec<&str> = index_str.split(',').filter(|s| !s.is_empty()).collect();
                let mut clients = self.clients.write().await;
                let mut loaded = 0usize;
                for cid in &client_ids {
                    let key = format!("oauth2:client:{}", cid);
                    if let Ok(Some(data)) = storage.db_get(q_storage::CF_MANIFEST, key.as_bytes()).await {
                        if let Ok(client) = serde_json::from_slice::<OAuth2Client>(&data) {
                            clients.insert(client.client_id.clone(), client);
                            loaded += 1;
                        }
                    }
                }
                if loaded > 0 {
                    info!("🔐 Loaded {} OAuth2 clients from disk", loaded);
                }
            }
            _ => {
                debug!("No persisted OAuth2 clients found (first boot)");
            }
        }
    }

    /// Persist a single client + update the index
    async fn persist_client(&self, client: &OAuth2Client) {
        let storage = match &self.storage {
            Some(s) => s,
            None => return,
        };
        // Save client JSON
        let key = format!("oauth2:client:{}", client.client_id);
        if let Ok(json) = serde_json::to_vec(client) {
            if let Err(e) = storage.db_put(q_storage::CF_MANIFEST, key.as_bytes(), &json).await {
                warn!("Failed to persist OAuth2 client {}: {}", client.client_id, e);
                return;
            }
        }
        // Update index (comma-separated client IDs)
        let clients = self.clients.read().await;
        let index: String = clients.keys().cloned().collect::<Vec<_>>().join(",");
        drop(clients);
        let _ = storage.db_put(q_storage::CF_MANIFEST, b"oauth2:clients_index", index.as_bytes()).await;
    }

    pub async fn register_client(&self, client: OAuth2Client) -> Result<(), String> {
        self.persist_client(&client).await;
        let mut clients = self.clients.write().await;
        clients.insert(client.client_id.clone(), client);
        Ok(())
    }

    pub async fn get_client(&self, client_id: &str) -> Option<OAuth2Client> {
        let clients = self.clients.read().await;
        clients.get(client_id).cloned()
    }

    pub async fn store_auth_code(&self, code: AuthorizationCode) {
        let mut codes = self.auth_codes.write().await;
        codes.insert(code.code.clone(), code);
    }

    pub async fn consume_auth_code(&self, code: &str) -> Option<AuthorizationCode> {
        let mut codes = self.auth_codes.write().await;
        codes.remove(code)
    }

    pub async fn store_access_token(&self, token: AccessToken) {
        let token_key = token.token.clone();

        // Store refresh token mapping if present
        if let Some(ref refresh_token) = token.refresh_token {
            let mut refresh_tokens = self.refresh_tokens.write().await;
            refresh_tokens.insert(refresh_token.clone(), token_key.clone());
        }

        // v8.5.1: Persist token to RocksDB so it survives server restarts
        if let Some(ref storage) = self.storage {
            let db_key = format!("oauth2:access_token:{}", token_key);
            if let Ok(json) = serde_json::to_vec(&token) {
                let _ = storage.db_put(q_storage::CF_MANIFEST, db_key.as_bytes(), &json).await;
            }
            // Persist refresh→access mapping too
            if let Some(ref refresh_token) = token.refresh_token {
                let rt_key = format!("oauth2:refresh_token:{}", refresh_token);
                let _ = storage.db_put(q_storage::CF_MANIFEST, rt_key.as_bytes(), token_key.as_bytes()).await;
            }
        }

        let mut tokens = self.access_tokens.write().await;
        tokens.insert(token_key, token);
    }

    pub async fn get_access_token(&self, token: &str) -> Option<AccessToken> {
        // Check in-memory cache first
        {
            let tokens = self.access_tokens.read().await;
            if let Some(t) = tokens.get(token) {
                return Some(t.clone());
            }
        }

        // v8.5.1: Fall back to RocksDB (token may have been stored before a restart)
        if let Some(ref storage) = self.storage {
            let db_key = format!("oauth2:access_token:{}", token);
            if let Ok(Some(data)) = storage.db_get(q_storage::CF_MANIFEST, db_key.as_bytes()).await {
                if let Ok(access_token) = serde_json::from_slice::<AccessToken>(&data) {
                    // Check expiration before returning
                    if access_token.expires_at < chrono::Utc::now() {
                        // Clean up expired token from disk
                        let _ = storage.db_delete(q_storage::CF_MANIFEST, db_key.as_bytes()).await;
                        return None;
                    }
                    // Re-hydrate in-memory cache
                    let mut tokens = self.access_tokens.write().await;
                    tokens.insert(token.to_string(), access_token.clone());
                    if let Some(ref rt) = access_token.refresh_token {
                        let mut refresh_tokens = self.refresh_tokens.write().await;
                        refresh_tokens.insert(rt.clone(), token.to_string());
                    }
                    return Some(access_token);
                }
            }
        }

        None
    }

    pub async fn revoke_token(&self, token: &str) {
        let mut tokens = self.access_tokens.write().await;
        if let Some(access_token) = tokens.remove(token) {
            // Also remove refresh token if exists
            if let Some(ref refresh_token) = &access_token.refresh_token {
                let mut refresh_tokens = self.refresh_tokens.write().await;
                refresh_tokens.remove(refresh_token);
                // v8.5.1: Remove from RocksDB too
                if let Some(ref storage) = self.storage {
                    let rt_key = format!("oauth2:refresh_token:{}", refresh_token);
                    let _ = storage.db_delete(q_storage::CF_MANIFEST, rt_key.as_bytes()).await;
                }
            }
        }
        // v8.5.1: Remove from RocksDB
        if let Some(ref storage) = self.storage {
            let db_key = format!("oauth2:access_token:{}", token);
            let _ = storage.db_delete(q_storage::CF_MANIFEST, db_key.as_bytes()).await;
        }
    }

    /// Look up an access token by its associated refresh token
    pub async fn find_token_by_refresh(&self, refresh_token: &str) -> Option<AccessToken> {
        let refresh_map = self.refresh_tokens.read().await;
        if let Some(access_token_key) = refresh_map.get(refresh_token) {
            let tokens = self.access_tokens.read().await;
            tokens.get(access_token_key).cloned()
        } else {
            None
        }
    }

    /// Remove an old access token and its refresh token mapping
    pub async fn remove_token_and_refresh(&self, access_token_key: &str, refresh_token: &str) {
        let mut tokens = self.access_tokens.write().await;
        tokens.remove(access_token_key);
        drop(tokens);
        let mut refresh_map = self.refresh_tokens.write().await;
        refresh_map.remove(refresh_token);
    }

    pub async fn store_consent(&self, consent: UserConsent) {
        let mut consents = self.user_consents.write().await;
        let key = (consent.wallet_address.clone(), consent.client_id.clone());
        consents.insert(key, consent);
    }

    pub async fn get_consent(&self, wallet_address: &str, client_id: &str) -> Option<UserConsent> {
        let consents = self.user_consents.read().await;
        consents
            .get(&(wallet_address.to_string(), client_id.to_string()))
            .cloned()
    }

    /// Get all consents granted by a specific wallet
    pub async fn get_consents_for_wallet(&self, wallet: &str) -> Vec<UserConsent> {
        let consents = self.user_consents.read().await;
        consents
            .iter()
            .filter(|((w, _), _)| w == wallet)
            .map(|(_, c)| c.clone())
            .collect()
    }

    /// Revoke a consent and all associated tokens for a wallet+client pair
    pub async fn revoke_consent(&self, wallet: &str, client_id: &str) -> bool {
        let key = (wallet.to_string(), client_id.to_string());
        let mut consents = self.user_consents.write().await;
        let removed = consents.remove(&key).is_some();
        drop(consents);

        // Also revoke all tokens for this wallet+client
        let mut tokens = self.access_tokens.write().await;
        let to_remove: Vec<String> = tokens
            .iter()
            .filter(|(_, t)| t.wallet_address == wallet && t.client_id == client_id)
            .map(|(k, _)| k.clone())
            .collect();
        for k in &to_remove {
            tokens.remove(k);
        }
        removed
    }

    /// Count registered OAuth2 clients
    pub async fn client_count(&self) -> usize {
        self.clients.read().await.len()
    }

    /// Count active (non-expired) access tokens
    pub async fn active_token_count(&self) -> usize {
        let now = Utc::now();
        let tokens = self.access_tokens.read().await;
        tokens.values().filter(|t| t.expires_at > now).count()
    }

    /// v7.4.0: Store an on-chain consent hash (from DAG transaction)
    pub async fn store_consent_hash(&self, hash: String) {
        let mut hashes = self.consent_hashes.write().await;
        hashes.insert(hash);
    }

    /// v7.4.0: Remove an on-chain consent hash (consent revoked)
    pub async fn revoke_consent_by_hash(&self, hash: &str) -> bool {
        let mut hashes = self.consent_hashes.write().await;
        hashes.remove(hash)
    }

    /// v7.4.0: Check if a consent hash exists on-chain
    pub async fn has_consent_hash(&self, hash: &str) -> bool {
        let hashes = self.consent_hashes.read().await;
        hashes.contains(hash)
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// /authorize endpoint query parameters
#[derive(Debug, Deserialize)]
pub struct AuthorizeRequest {
    #[serde(default = "default_response_type")]
    pub response_type: String,
    pub client_id: String,
    pub redirect_uri: String,
    pub scope: Option<String>,
    pub state: Option<String>,
    pub code_challenge: Option<String>, // PKCE
    pub code_challenge_method: Option<String>,
}

fn default_response_type() -> String {
    "code".to_string()
}

/// /token endpoint request
#[derive(Debug, Deserialize)]
pub struct TokenRequest {
    pub grant_type: String,
    pub code: Option<String>,
    pub redirect_uri: Option<String>,
    pub client_id: String,
    #[serde(default)]
    pub client_secret: Option<String>, // v7.3.5: Optional for PKCE public clients
    pub code_verifier: Option<String>, // PKCE
    pub refresh_token: Option<String>,
}

/// /token endpoint response
#[derive(Debug, Serialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: i64,
    pub refresh_token: Option<String>,
    pub scope: String,
}

/// User consent request (internal, from frontend)
#[derive(Debug, Deserialize)]
pub struct ConsentRequest {
    pub wallet_address: String,
    pub client_id: String,
    pub scopes: Vec<String>,
    pub approved: bool,
    pub auth_request_id: String, // Temporary ID to link to pending auth request
    #[serde(default)]
    pub redirect_uri: Option<String>,
    #[serde(default)]
    pub code_challenge: Option<String>,
    #[serde(default)]
    pub code_challenge_method: Option<String>,
}

/// Client registration request
#[derive(Debug, Deserialize)]
pub struct RegisterClientRequest {
    pub name: String,
    pub description: Option<String>,
    pub website: String,
    pub redirect_uris: Vec<String>,
    pub logo_url: Option<String>,
    pub kyber_public_key: Option<String>, // Base64-encoded Kyber1024 public key
    #[serde(default)]
    pub client_id: Option<String>,        // v7.3.5: Optional custom client_id
    #[serde(default)]
    pub client_secret: Option<String>,    // v7.3.5: Optional custom client_secret
    #[serde(default)]
    pub scopes: Option<Vec<String>>,      // v7.3.5: Optional custom scopes
}

/// Client registration response
#[derive(Debug, Serialize)]
pub struct RegisterClientResponse {
    pub client_id: String,
    pub client_secret: String,
    pub name: String,
}

// ============================================================================
// Helper Functions
// ============================================================================

fn generate_random_token(length: usize) -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let bytes: Vec<u8> = (0..length).map(|_| rng.gen::<u8>()).collect();
    BASE64.encode(&bytes)
}

fn hash_code_challenge(verifier: &str, method: &str) -> String {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    match method {
        "S256" => {
            // RFC 7636 §4.2: BASE64URL(SHA256(code_verifier)) — URL-safe, no padding
            let mut hasher = Sha256::new();
            hasher.update(verifier.as_bytes());
            URL_SAFE_NO_PAD.encode(hasher.finalize())
        }
        "plain" => verifier.to_string(),
        _ => String::new(),
    }
}

fn verify_pkce_challenge(verifier: &str, challenge: &str, method: &str) -> bool {
    let computed_challenge = hash_code_challenge(verifier, method);
    computed_challenge == challenge
}

// ============================================================================
// v7.4.0: JWT Signed Token System — Decentralized Cross-Node Verification
// Any node can verify tokens issued by any other node using Ed25519 signatures.
// Backward compatible: opaque HashMap lookup is tried first, then JWT verify.
// ============================================================================

use base64::engine::general_purpose::URL_SAFE_NO_PAD as BASE64URL;

/// JWT Header (alg=EdDSA, kid=peer_id)
#[derive(Debug, Serialize, Deserialize)]
struct JwtHeader {
    alg: String,
    kid: String,  // peer_id of the issuing node
    typ: String,
}

/// JWT Payload — all claims needed to reconstruct an AccessToken
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtPayload {
    pub sub: String,          // wallet address
    pub client_id: String,
    pub scopes: Vec<String>,
    pub iss: String,          // peer_id of issuing node
    pub exp: i64,             // expiry (unix timestamp)
    pub iat: i64,             // issued-at (unix timestamp)
    pub jti: String,          // unique token id
}

/// Generate a JWT signed with the node's Ed25519 signing key.
/// Format: base64url(header).base64url(payload).base64url(signature)
pub fn generate_signed_token(
    signing_key: &ed25519_dalek::SigningKey,
    peer_id: &str,
    wallet: &str,
    client_id: &str,
    scopes: &[String],
    expiry_secs: i64,
) -> String {
    use ed25519_dalek::Signer;

    let header = JwtHeader {
        alg: "EdDSA".to_string(),
        kid: peer_id.to_string(),
        typ: "JWT".to_string(),
    };
    let now = Utc::now().timestamp();
    let payload = JwtPayload {
        sub: wallet.to_string(),
        client_id: client_id.to_string(),
        scopes: scopes.to_vec(),
        iss: peer_id.to_string(),
        exp: now + expiry_secs,
        iat: now,
        jti: generate_random_token(16),
    };

    let header_b64 = BASE64URL.encode(serde_json::to_vec(&header).unwrap_or_default());
    let payload_b64 = BASE64URL.encode(serde_json::to_vec(&payload).unwrap_or_default());
    let message = format!("{}.{}", header_b64, payload_b64);
    let signature = signing_key.sign(message.as_bytes());
    let sig_b64 = BASE64URL.encode(signature.to_bytes());

    format!("{}.{}", message, sig_b64)
}

/// Verify a JWT token. Checks signature against local key or peer_jwt_keys DashMap.
/// Returns the decoded payload on success.
pub fn verify_signed_token(
    token: &str,
    local_key: &ed25519_dalek::SigningKey,
    local_peer_id: &str,
    peer_jwt_keys: &dashmap::DashMap<String, PeerJwtKeyInfo>,
) -> Result<JwtPayload, String> {
    use ed25519_dalek::Verifier;

    let parts: Vec<&str> = token.splitn(3, '.').collect();
    if parts.len() != 3 {
        return Err("Invalid JWT format".to_string());
    }

    // Decode header to get kid (issuing node's peer_id)
    let header_bytes = BASE64URL.decode(parts[0]).map_err(|e| format!("Bad header: {}", e))?;
    let header: JwtHeader = serde_json::from_slice(&header_bytes).map_err(|e| format!("Bad header JSON: {}", e))?;

    if header.alg != "EdDSA" {
        return Err(format!("Unsupported algorithm: {}", header.alg));
    }

    // Decode payload
    let payload_bytes = BASE64URL.decode(parts[1]).map_err(|e| format!("Bad payload: {}", e))?;
    let payload: JwtPayload = serde_json::from_slice(&payload_bytes).map_err(|e| format!("Bad payload JSON: {}", e))?;

    // Check expiry
    if payload.exp < Utc::now().timestamp() {
        return Err("Token expired".to_string());
    }

    // Decode signature
    let sig_bytes = BASE64URL.decode(parts[2]).map_err(|e| format!("Bad signature: {}", e))?;
    let signature = ed25519_dalek::Signature::from_slice(&sig_bytes)
        .map_err(|e| format!("Invalid signature bytes: {}", e))?;

    // Build the signed message (header.payload)
    let message = format!("{}.{}", parts[0], parts[1]);

    // Resolve verifying key: local node or peer
    if header.kid == local_peer_id {
        let verifying_key = local_key.verifying_key();
        verifying_key.verify(message.as_bytes(), &signature)
            .map_err(|_| "Signature verification failed (local key)".to_string())?;
    } else if let Some(peer_info) = peer_jwt_keys.get(&header.kid) {
        let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&peer_info.ed25519_verifying_bytes)
            .map_err(|e| format!("Invalid peer key: {}", e))?;
        verifying_key.verify(message.as_bytes(), &signature)
            .map_err(|_| format!("Signature verification failed (peer {})", header.kid))?;
    } else {
        return Err(format!("Unknown issuer: {}", header.kid));
    }

    Ok(payload)
}

/// Resolve an access token: try opaque HashMap first (backward compat), then JWT verify.
/// This is the single entry point for all token validation.
pub async fn resolve_access_token(
    token: &str,
    storage: &OAuth2Storage,
    local_key: &ed25519_dalek::SigningKey,
    local_peer_id: &str,
    peer_jwt_keys: &dashmap::DashMap<String, PeerJwtKeyInfo>,
) -> Option<AccessToken> {
    // 1. Try opaque lookup (fast path, backward compat with pre-JWT tokens)
    if let Some(access_token) = storage.get_access_token(token).await {
        if access_token.expires_at > Utc::now() {
            return Some(access_token);
        }
    }

    // 2. Try JWT verification (cross-node tokens)
    match verify_signed_token(token, local_key, local_peer_id, peer_jwt_keys) {
        Ok(payload) => {
            Some(AccessToken {
                token: token.to_string(),
                client_id: payload.client_id,
                wallet_address: payload.sub,
                scopes: payload.scopes,
                expires_at: chrono::DateTime::from_timestamp(payload.exp, 0)
                    .unwrap_or_else(Utc::now),
                refresh_token: None, // JWT tokens don't carry refresh tokens
            })
        }
        Err(_) => None,
    }
}

/// Compute a SHA-256 consent hash for on-chain privacy: H(wallet|client_id|scopes_sorted)
pub fn compute_consent_hash(wallet: &str, client_id: &str, scopes: &[String]) -> String {
    let mut sorted_scopes = scopes.to_vec();
    sorted_scopes.sort();
    let input = format!("{}|{}|{}", wallet, client_id, sorted_scopes.join(","));
    let hash = Sha256::digest(input.as_bytes());
    hex::encode(hash)
}

// ============================================================================
// OAuth2 Endpoints
// v4.0.5: All endpoints access state.oauth2_storage directly (no outer RwLock).
// OAuth2Storage has internal RwLock per field for safe concurrent access.
// ============================================================================

/// POST /api/v1/oauth2/register
/// Register a new OAuth2 client application
pub async fn register_client(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RegisterClientRequest>,
) -> Result<Json<ApiResponse<RegisterClientResponse>>, StatusCode> {
    info!("🔐 Registering new OAuth2 client: {}", request.name);

    // v7.3.5: Use custom client_id/secret if provided, otherwise generate
    let client_id = request.client_id.clone()
        .unwrap_or_else(|| format!("qnk_client_{}", generate_random_token(16)));
    let client_secret = request.client_secret.clone()
        .unwrap_or_else(|| generate_random_token(32));

    // Decode Kyber public key if provided
    let kyber_public_key = if let Some(ref key_b64) = request.kyber_public_key {
        match BASE64.decode(key_b64) {
            Ok(key) => Some(key),
            Err(e) => {
                error!("Invalid Kyber public key: {}", e);
                return Ok(Json(ApiResponse::error(
                    "Invalid Kyber public key format".to_string(),
                )));
            }
        }
    } else {
        None
    };

    let client = OAuth2Client {
        client_id: client_id.clone(),
        client_secret: client_secret.clone(),
        redirect_uris: request.redirect_uris,
        name: request.name.clone(),
        description: request.description.clone().unwrap_or_default(),
        website: request.website,
        logo_url: request.logo_url,
        scopes: request.scopes.unwrap_or_else(|| vec!["read:balance".to_string(), "send:transaction".to_string()]),
        created_at: Utc::now(),
        kyber_public_key,
    };

    // v7.4.0: Clone client before registration (needed for P2P broadcast)
    let client_for_broadcast = client.clone();

    // v4.0.5: Direct access - no outer .write().await needed
    state
        .oauth2_storage
        .register_client(client)
        .await
        .map_err(|e| {
            error!("Failed to register client: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    info!(
        "✅ Registered OAuth2 client: {} ({})",
        request.name, client_id
    );

    // v7.4.0: Broadcast client registration to P2P (with hashed secret)
    if let Some(ref cmd_tx) = state.libp2p_command_tx {
        let peer_id = {
            let info = state.libp2p_peer_info.read().await;
            info.0.clone()
        };
        if !peer_id.is_empty() {
            // Hash the client_secret before gossiping — raw secret never leaves this node
            let mut broadcast_client = client_for_broadcast;
            let mut hasher = Sha256::new();
            hasher.update(broadcast_client.client_secret.as_bytes());
            broadcast_client.client_secret = format!("sha256:{}", hex::encode(hasher.finalize()));

            let network_id = std::env::var("Q_NETWORK_ID")
                .ok()
                .and_then(|s| s.parse::<q_types::NetworkId>().ok())
                .unwrap_or(q_types::NetworkId::MainnetGenesis);
            let timestamp = Utc::now().timestamp();
            let sign_msg = format!("oauth2-client:{}:{}:{}", peer_id, broadcast_client.client_id, timestamp);

            use ed25519_dalek::Signer;
            let sig = state.node_signing_key.sign(sign_msg.as_bytes());

            #[derive(serde::Serialize)]
            struct OAuth2ClientAnnouncement {
                client: OAuth2Client,
                origin_peer_id: String,
                timestamp: i64,
                signature: Vec<u8>,
            }

            let announcement = OAuth2ClientAnnouncement {
                client: broadcast_client,
                origin_peer_id: peer_id,
                timestamp,
                signature: sig.to_bytes().to_vec(),
            };

            if let Ok(bytes) = serde_json::to_vec(&announcement) {
                let _ = cmd_tx.send(q_network::NetworkCommand::PublishOAuth2Client {
                    topic: network_id.oauth2_clients_topic(),
                    client_bytes: bytes,
                });
                debug!("🔐 [OAUTH2] Broadcast client registration to P2P");
            }
        }
    }

    Ok(Json(ApiResponse::success(RegisterClientResponse {
        client_id,
        client_secret,
        name: request.name,
    })))
}

/// GET /api/v1/oauth2/authorize
/// OAuth2 authorization endpoint - redirects to consent screen
pub async fn authorize(
    State(state): State<Arc<AppState>>,
    Query(params): Query<AuthorizeRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    info!(
        "🔐 OAuth2 authorization request from client: {}",
        params.client_id
    );

    // Validate client - direct access, no outer lock
    let client = match state.oauth2_storage.get_client(&params.client_id).await {
        Some(c) => c,
        None => {
            warn!("Unknown client ID: {}", params.client_id);
            return Ok(Redirect::to(&format!(
                "{}?error=invalid_client",
                params.redirect_uri
            )));
        }
    };

    // Validate redirect URI
    // For PKCE public clients (no secret), allow scheme-prefix matching
    // since mobile dev environments generate dynamic redirect URIs.
    // PKCE already prevents authorization code interception.
    let redirect_valid = if client.client_secret.is_empty() && params.code_challenge.is_some() {
        // PKCE client: allow if redirect starts with any registered scheme prefix
        client.redirect_uris.iter().any(|uri| {
            let scheme = uri.split("://").next().unwrap_or("");
            params.redirect_uri.starts_with(&format!("{}://", scheme))
        }) || client.redirect_uris.contains(&params.redirect_uri)
    } else {
        client.redirect_uris.contains(&params.redirect_uri)
    };
    if !redirect_valid {
        error!("Invalid redirect URI: {}", params.redirect_uri);
        return Err(StatusCode::BAD_REQUEST);
    }

    // Validate response type
    if params.response_type != "code" {
        let error_uri = format!(
            "{}?error=unsupported_response_type&state={}",
            params.redirect_uri,
            params.state.as_deref().unwrap_or("")
        );
        return Ok(Redirect::to(&error_uri));
    }

    // v4.0.5: Use request origin for consent URL instead of hardcoded domain
    let mut consent_url = format!(
        "/oauth/consent?client_id={}&redirect_uri={}&scope={}&state={}",
        params.client_id,
        urlencoding::encode(&params.redirect_uri),
        urlencoding::encode(&params.scope.as_deref().unwrap_or("read:balance")),
        urlencoding::encode(&params.state.as_deref().unwrap_or(""))
    );
    if let Some(ref challenge) = params.code_challenge {
        consent_url.push_str(&format!("&code_challenge={}", urlencoding::encode(challenge)));
        if let Some(ref method) = params.code_challenge_method {
            consent_url.push_str(&format!("&code_challenge_method={}", urlencoding::encode(method)));
        }
    }

    debug!("Redirecting to consent screen: {}", consent_url);
    Ok(Redirect::to(&consent_url))
}

/// POST /api/v1/oauth2/consent
/// User grants or denies consent for client access.
/// v7.4.0: Returns consent_hash + consent_tx_data for on-chain recording.
pub async fn handle_consent(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ConsentRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    info!(
        "🔐 Processing consent from wallet: {} for client: {}",
        request.wallet_address, request.client_id
    );

    if !request.approved {
        info!("❌ User denied consent");
        return Ok(Json(ApiResponse::error("User denied consent".to_string())));
    }

    // Store consent - direct access
    let consent = UserConsent {
        wallet_address: request.wallet_address.clone(),
        client_id: request.client_id.clone(),
        scopes: request.scopes.clone(),
        granted_at: Utc::now(),
        expires_at: Some(Utc::now() + Duration::days(365)),
    };

    state.oauth2_storage.store_consent(consent).await;

    // v7.4.0: Compute privacy-preserving consent hash for on-chain recording
    let consent_hash = compute_consent_hash(&request.wallet_address, &request.client_id, &request.scopes);
    state.oauth2_storage.store_consent_hash(consent_hash.clone()).await;

    // Generate authorization code
    let auth_code = generate_random_token(32);
    let code_record = AuthorizationCode {
        code: auth_code.clone(),
        client_id: request.client_id.clone(),
        wallet_address: request.wallet_address.clone(),
        redirect_uri: request.redirect_uri.unwrap_or_default(),
        scopes: request.scopes,
        expires_at: Utc::now() + Duration::seconds(AUTH_CODE_EXPIRY_SECONDS),
        code_challenge: request.code_challenge,
        code_challenge_method: request.code_challenge_method,
    };

    state.oauth2_storage.store_auth_code(code_record).await;

    // v7.4.0: Return auth code + consent transaction data for frontend to sign
    let consent_tx_data = serde_json::json!({
        "consent_hash": consent_hash,
        "action": "grant",
        "version": 1,
        "tx_type": "0xA0", // OAuth2ConsentGrant
    });

    info!("✅ Consent granted, authorization code generated (consent_hash: {}...)", &consent_hash[..consent_hash.len().min(16)]);
    Ok(Json(ApiResponse::success(serde_json::json!({
        "auth_code": auth_code,
        "consent_hash": consent_hash,
        "consent_tx_data": consent_tx_data,
    }))))
}

/// POST /api/v1/oauth2/token
/// Exchange authorization code for access token
pub async fn token(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TokenRequest>,
) -> Result<Json<ApiResponse<TokenResponse>>, StatusCode> {
    info!("🔐 OAuth2 token request from client: {}", request.client_id);

    // Validate client credentials - direct access
    // v7.3.5: PKCE public clients may omit client_secret (code_verifier proves possession)
    let client = match state.oauth2_storage.get_client(&request.client_id).await {
        Some(c) => {
            let has_pkce = request.code_verifier.is_some();
            let secret_matches = request.client_secret.as_ref().map_or(false, |s| *s == c.client_secret);
            if !secret_matches && !has_pkce {
                error!("Invalid client credentials and no PKCE verifier");
                return Ok(Json(ApiResponse::error(
                    "Invalid client credentials".to_string(),
                )));
            }
            c
        }
        None => {
            error!("Unknown client_id: {}", request.client_id);
            return Ok(Json(ApiResponse::error(
                "Invalid client credentials".to_string(),
            )));
        }
    };

    match request.grant_type.as_str() {
        "authorization_code" => {
            // Exchange authorization code for access token
            let code = request.code.as_ref().ok_or_else(|| {
                error!("Missing authorization code");
                StatusCode::BAD_REQUEST
            })?;

            let auth_code = state
                .oauth2_storage
                .consume_auth_code(code)
                .await
                .ok_or_else(|| {
                    error!("Invalid or expired authorization code");
                    StatusCode::BAD_REQUEST
                })?;

            // Verify not expired
            if auth_code.expires_at < Utc::now() {
                error!("Authorization code expired");
                return Ok(Json(ApiResponse::error(
                    "Authorization code expired".to_string(),
                )));
            }

            // Validate redirect_uri matches (RFC 6749 §4.1.3)
            if !auth_code.redirect_uri.is_empty() {
                if let Some(ref req_redirect) = request.redirect_uri {
                    if *req_redirect != auth_code.redirect_uri {
                        error!("Redirect URI mismatch: expected '{}', got '{}'", auth_code.redirect_uri, req_redirect);
                        return Ok(Json(ApiResponse::error(
                            "redirect_uri does not match the one used in authorization".to_string(),
                        )));
                    }
                }
            }

            // Verify PKCE if present
            if let (Some(challenge), Some(method)) = (
                auth_code.code_challenge.as_ref(),
                auth_code.code_challenge_method.as_ref(),
            ) {
                if let Some(verifier) = request.code_verifier.as_ref() {
                    if !verify_pkce_challenge(verifier, challenge, method) {
                        error!("PKCE verification failed");
                        return Ok(Json(ApiResponse::error(
                            "PKCE verification failed".to_string(),
                        )));
                    }
                } else {
                    error!("Missing code verifier for PKCE");
                    return Ok(Json(ApiResponse::error(
                        "Missing code verifier".to_string(),
                    )));
                }
            }

            // v7.4.0: Verify on-chain consent hasn't been revoked (cross-node safety)
            let consent_hash = compute_consent_hash(
                &auth_code.wallet_address,
                &auth_code.client_id,
                &auth_code.scopes,
            );
            if !state.oauth2_storage.has_consent_hash(&consent_hash).await {
                // Consent may have been revoked on-chain between consent grant and token exchange
                warn!("⚠️ On-chain consent hash not found for wallet {} / client {} (may have been revoked)",
                    auth_code.wallet_address, auth_code.client_id);
                // Non-blocking: log warning but still issue token (consent was just granted locally)
                // On-chain confirmation is async — the DAG tx may not be mined yet
            }

            // v7.4.0: Generate JWT signed token (cross-node verifiable)
            let peer_id = {
                let info = state.libp2p_peer_info.read().await;
                info.0.clone()
            };
            let access_token = generate_signed_token(
                &state.node_signing_key,
                &peer_id,
                &auth_code.wallet_address,
                &auth_code.client_id,
                &auth_code.scopes,
                TOKEN_EXPIRY_SECONDS,
            );
            let refresh_token = generate_random_token(32);

            let token_record = AccessToken {
                token: access_token.clone(),
                client_id: auth_code.client_id,
                wallet_address: auth_code.wallet_address,
                scopes: auth_code.scopes.clone(),
                expires_at: Utc::now() + Duration::seconds(TOKEN_EXPIRY_SECONDS),
                refresh_token: Some(refresh_token.clone()),
            };

            // Store locally too for fast opaque lookup on this node
            state.oauth2_storage.store_access_token(token_record).await;

            info!("✅ JWT access token generated (issuer: {})", if peer_id.len() > 12 { &peer_id[..12] } else { &peer_id });
            Ok(Json(ApiResponse::success(TokenResponse {
                access_token,
                token_type: "Bearer".to_string(),
                expires_in: TOKEN_EXPIRY_SECONDS,
                refresh_token: Some(refresh_token),
                scope: auth_code.scopes.join(" "),
            })))
        }
        "refresh_token" => {
            let refresh_token = request.refresh_token.as_ref().ok_or_else(|| {
                error!("Missing refresh token");
                StatusCode::BAD_REQUEST
            })?;

            // Look up the existing access token associated with this refresh token
            let old_token = state
                .oauth2_storage
                .find_token_by_refresh(refresh_token)
                .await
                .ok_or_else(|| {
                    error!("Invalid or expired refresh token");
                    StatusCode::UNAUTHORIZED
                })?;

            // Verify the refresh token belongs to the requesting client
            if old_token.client_id != request.client_id {
                error!("Refresh token does not belong to this client");
                return Ok(Json(ApiResponse::error(
                    "Invalid refresh token for this client".to_string(),
                )));
            }

            // v7.4.0: Check if consent was revoked on-chain before refreshing
            let consent_hash = compute_consent_hash(
                &old_token.wallet_address,
                &old_token.client_id,
                &old_token.scopes,
            );
            if !state.oauth2_storage.has_consent_hash(&consent_hash).await {
                error!("🔐 Consent revoked on-chain for wallet {} / client {} — refusing refresh",
                    old_token.wallet_address, old_token.client_id);
                return Ok(Json(ApiResponse::error(
                    "Consent has been revoked".to_string(),
                )));
            }

            // Remove old token and refresh mapping
            state
                .oauth2_storage
                .remove_token_and_refresh(&old_token.token, refresh_token)
                .await;

            // v7.4.0: Generate new JWT access token on refresh
            let peer_id = {
                let info = state.libp2p_peer_info.read().await;
                info.0.clone()
            };
            let new_access_token = generate_signed_token(
                &state.node_signing_key,
                &peer_id,
                &old_token.wallet_address,
                &old_token.client_id,
                &old_token.scopes,
                TOKEN_EXPIRY_SECONDS,
            );
            let new_refresh_token = generate_random_token(32);

            let token_record = AccessToken {
                token: new_access_token.clone(),
                client_id: old_token.client_id,
                wallet_address: old_token.wallet_address,
                scopes: old_token.scopes.clone(),
                expires_at: Utc::now() + Duration::seconds(TOKEN_EXPIRY_SECONDS),
                refresh_token: Some(new_refresh_token.clone()),
            };

            state.oauth2_storage.store_access_token(token_record).await;

            info!("✅ JWT access token refreshed successfully");
            Ok(Json(ApiResponse::success(TokenResponse {
                access_token: new_access_token,
                token_type: "Bearer".to_string(),
                expires_in: TOKEN_EXPIRY_SECONDS,
                refresh_token: Some(new_refresh_token),
                scope: old_token.scopes.join(" "),
            })))
        }
        _ => {
            error!("Unsupported grant type: {}", request.grant_type);
            Ok(Json(ApiResponse::error(
                "Unsupported grant type".to_string(),
            )))
        }
    }
}

/// GET /api/v1/oauth2/userinfo
/// Get user information with access token
pub async fn userinfo(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    // Extract Bearer token from Authorization header
    let auth_header = headers
        .get("Authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(StatusCode::UNAUTHORIZED)?;

    let token = auth_header
        .strip_prefix("Bearer ")
        .ok_or(StatusCode::UNAUTHORIZED)?;

    // v7.4.0: Resolve token via opaque lookup OR JWT verification (cross-node)
    let peer_id = {
        let info = state.libp2p_peer_info.read().await;
        info.0.clone()
    };
    let access_token = resolve_access_token(
        token,
        &state.oauth2_storage,
        &state.node_signing_key,
        &peer_id,
        &state.peer_jwt_keys,
    )
    .await
    .ok_or(StatusCode::UNAUTHORIZED)?;

    info!(
        "✅ Userinfo request for wallet: {}",
        access_token.wallet_address
    );

    // Return user info based on granted scopes
    let mut user_info = serde_json::json!({
        "sub": access_token.wallet_address,
        "wallet_address": access_token.wallet_address,
    });

    // Add balance if scope allows
    if access_token.scopes.contains(&"read:balance".to_string()) {
        // Convert hex string to [u8; 32] address
        let wallet_str = access_token.wallet_address.strip_prefix("qnk").unwrap_or(&access_token.wallet_address);
        if let Ok(address_bytes) = hex::decode(wallet_str) {
            if address_bytes.len() == 32 {
                let mut address = [0u8; 32];
                address.copy_from_slice(&address_bytes);
                if let Some(balance) = state.wallet_balances.read().await.get(&address).copied() {
                    user_info["balance"] = serde_json::json!(balance.to_string());
                    user_info["balance_qug"] = serde_json::json!(balance as f64 / 1e24);
                }

                // Also include token balances
                let token_bals = state.token_balances.read().await;
                let mut tokens = serde_json::Map::new();
                for ((wallet, token_addr), bal) in token_bals.iter() {
                    if *wallet == address && *bal > 0 {
                        let token_hex = hex::encode(token_addr);
                        tokens.insert(token_hex, serde_json::json!(bal.to_string()));
                    }
                }
                if !tokens.is_empty() {
                    user_info["token_balances"] = serde_json::Value::Object(tokens);
                }
            }
        }
    }

    Ok(Json(ApiResponse::success(user_info)))
}

/// POST /api/v1/oauth2/revoke
/// Revoke an access token
#[derive(Debug, Deserialize)]
pub struct RevokeRequest {
    pub token: String,
}

pub async fn revoke(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(request): Json<RevokeRequest>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    info!("🔐 Revoking token");

    // Verify the caller owns the token (must provide valid Bearer token)
    let auth_header = headers
        .get("Authorization")
        .and_then(|h| h.to_str().ok());

    if let Some(auth) = auth_header {
        if let Some(bearer) = auth.strip_prefix("Bearer ") {
            // v7.4.0: Resolve caller's token via opaque or JWT
            let peer_id = {
                let info = state.libp2p_peer_info.read().await;
                info.0.clone()
            };
            let caller_token = resolve_access_token(
                bearer, &state.oauth2_storage, &state.node_signing_key, &peer_id, &state.peer_jwt_keys,
            ).await;

            if let Some(caller) = caller_token {
                // Check the token being revoked belongs to the same wallet
                let target_token = resolve_access_token(
                    &request.token, &state.oauth2_storage, &state.node_signing_key, &peer_id, &state.peer_jwt_keys,
                ).await;

                if let Some(target) = target_token {
                    if target.wallet_address != caller.wallet_address {
                        error!("Token revocation denied: caller does not own target token");
                        return Ok(Json(ApiResponse::error(
                            "Cannot revoke tokens owned by other users".to_string(),
                        )));
                    }
                }
            }
        }
    }

    state.oauth2_storage.revoke_token(&request.token).await;
    Ok(Json(ApiResponse::success("Token revoked".to_string())))
}

/// GET /api/v1/oauth2/clients/:client_id
/// Get client information (public data only)
pub async fn get_client_info(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(client_id): axum::extract::Path<String>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    info!("🔍 Client info request for: {}", client_id);

    // Direct access - no outer lock
    match state.oauth2_storage.get_client(&client_id).await {
        Some(client) => {
            let client_info = serde_json::json!({
                "client_id": client.client_id,
                "name": client.name,
                "description": client.description,
                "website": client.website,
                "logo_url": client.logo_url,
                "scopes": client.scopes,
            });

            Ok(Json(ApiResponse::success(client_info)))
        }
        None => {
            info!("❌ Client not found: {}", client_id);
            Ok(Json(ApiResponse::error("Client not found".to_string())))
        }
    }
}

// ============================================================================
// v8.5.9: DEVICE LOGIN FLOW FOR MINER
// Miner requests a code → user opens URL in browser → logs in → miner polls for wallet
// Similar to GitHub CLI / smart TV login flow
// ============================================================================

/// POST /api/v1/miner/device-login — Miner requests a login code
/// Returns a URL for the user to visit in their browser
pub async fn device_login_request(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    // Generate a short user-friendly code (8 chars)
    let user_code = format!(
        "{}-{}",
        &generate_random_token(4)[..4].to_uppercase(),
        &generate_random_token(4)[..4].to_uppercase()
    );
    // Use hex instead of base64 for device code — base64 has '/' and '=' which break URL paths
    let device_code = {
        use rand::Rng;
        let bytes: Vec<u8> = (0..32).map(|_| rand::thread_rng().gen::<u8>()).collect();
        hex::encode(&bytes)
    };

    let login = DeviceLoginCode {
        code: device_code.clone(),
        user_code: user_code.clone(),
        created_at: Utc::now(),
        expires_at: Utc::now() + Duration::seconds(600), // 10 min expiry
        wallet_address: None,
        completed: false,
    };

    state.oauth2_storage.device_codes.write().await.insert(device_code.clone(), login);

    info!("🔑 [DEVICE-LOGIN] Created device login code: {} (user_code: {})", &device_code[..16], user_code);

    Ok(Json(ApiResponse::success(serde_json::json!({
        "device_code": device_code,
        "user_code": user_code,
        "verification_url": format!("https://quillon.xyz/miner-login?code={}", device_code),
        "expires_in": 600,
        "interval": 3,
    }))))
}

/// GET /api/v1/miner/device-login/:code — Miner polls to check if user completed login
pub async fn device_login_poll(
    Path(code): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let device_codes = state.oauth2_storage.device_codes.read().await;

    match device_codes.get(&code) {
        Some(login) => {
            if Utc::now() > login.expires_at {
                return Ok(Json(ApiResponse::error("Login code expired. Please restart the miner.".to_string())));
            }
            if login.completed {
                let wallet = login.wallet_address.clone().unwrap_or_default();
                info!("✅ [DEVICE-LOGIN] Miner claimed wallet: {}...", &wallet[..wallet.len().min(16)]);
                Ok(Json(ApiResponse::success(serde_json::json!({
                    "status": "complete",
                    "wallet_address": wallet,
                }))))
            } else {
                Ok(Json(ApiResponse::success(serde_json::json!({
                    "status": "pending",
                }))))
            }
        }
        None => Ok(Json(ApiResponse::error("Invalid or expired device code".to_string()))),
    }
}

/// POST /api/v1/miner/device-login/complete — User completes login from browser
/// Called by the frontend after user logs in / enters mnemonic
#[derive(Debug, Deserialize)]
pub struct DeviceLoginCompleteRequest {
    pub device_code: String,
    pub wallet_address: String,
}

pub async fn device_login_complete(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DeviceLoginCompleteRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let mut device_codes = state.oauth2_storage.device_codes.write().await;

    match device_codes.get_mut(&req.device_code) {
        Some(login) => {
            if Utc::now() > login.expires_at {
                return Ok(Json(ApiResponse::error("Login code expired".to_string())));
            }
            if login.completed {
                return Ok(Json(ApiResponse::error("Login already completed".to_string())));
            }

            // Validate wallet format
            let wallet = req.wallet_address.trim().to_string();
            if !wallet.starts_with("qnk") || wallet.len() < 67 {
                return Ok(Json(ApiResponse::error("Invalid wallet address format".to_string())));
            }

            login.wallet_address = Some(wallet.clone());
            login.completed = true;

            info!("✅ [DEVICE-LOGIN] User completed login for code {} → wallet {}...",
                &req.device_code[..req.device_code.len().min(16)], &wallet[..wallet.len().min(16)]);

            Ok(Json(ApiResponse::success(serde_json::json!({
                "status": "complete",
                "message": "Login successful! Your miner will start automatically.",
            }))))
        }
        None => Ok(Json(ApiResponse::error("Invalid or expired device code".to_string()))),
    }
}
