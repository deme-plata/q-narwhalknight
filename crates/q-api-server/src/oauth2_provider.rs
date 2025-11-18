// OAuth2 Provider for Quillon Wallet - Third-party Integration
// Allows external websites to authenticate users and request wallet operations
// Uses post-quantum encryption (Kyber1024) for all sensitive data

use axum::{
    extract::{Json, Query, State},
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

// ============================================================================
// OAuth2 Storage
// ============================================================================

pub struct OAuth2Storage {
    clients: RwLock<HashMap<String, OAuth2Client>>,
    auth_codes: RwLock<HashMap<String, AuthorizationCode>>,
    access_tokens: RwLock<HashMap<String, AccessToken>>,
    refresh_tokens: RwLock<HashMap<String, String>>, // refresh_token -> access_token
    user_consents: RwLock<HashMap<(String, String), UserConsent>>, // (wallet, client_id) -> consent
}

impl OAuth2Storage {
    pub fn new() -> Self {
        Self {
            clients: RwLock::new(HashMap::new()),
            auth_codes: RwLock::new(HashMap::new()),
            access_tokens: RwLock::new(HashMap::new()),
            refresh_tokens: RwLock::new(HashMap::new()),
            user_consents: RwLock::new(HashMap::new()),
        }
    }

    pub async fn register_client(&self, client: OAuth2Client) -> Result<(), String> {
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
        let mut tokens = self.access_tokens.write().await;

        // Store refresh token mapping if present
        if let Some(ref refresh_token) = token.refresh_token {
            let mut refresh_tokens = self.refresh_tokens.write().await;
            refresh_tokens.insert(refresh_token.clone(), token_key.clone());
        }

        tokens.insert(token_key, token);
    }

    pub async fn get_access_token(&self, token: &str) -> Option<AccessToken> {
        let tokens = self.access_tokens.read().await;
        tokens.get(token).cloned()
    }

    pub async fn revoke_token(&self, token: &str) {
        let mut tokens = self.access_tokens.write().await;
        if let Some(access_token) = tokens.remove(token) {
            // Also remove refresh token if exists
            if let Some(refresh_token) = access_token.refresh_token {
                let mut refresh_tokens = self.refresh_tokens.write().await;
                refresh_tokens.remove(&refresh_token);
            }
        }
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
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// /authorize endpoint query parameters
#[derive(Debug, Deserialize)]
pub struct AuthorizeRequest {
    pub response_type: String,
    pub client_id: String,
    pub redirect_uri: String,
    pub scope: Option<String>,
    pub state: Option<String>,
    pub code_challenge: Option<String>, // PKCE
    pub code_challenge_method: Option<String>,
}

/// /token endpoint request
#[derive(Debug, Deserialize)]
pub struct TokenRequest {
    pub grant_type: String,
    pub code: Option<String>,
    pub redirect_uri: Option<String>,
    pub client_id: String,
    pub client_secret: String,
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
    let bytes: Vec<u8> = (0..length).map(|_| rng.gen()).collect();
    BASE64.encode(&bytes)
}

fn hash_code_challenge(verifier: &str, method: &str) -> String {
    match method {
        "S256" => {
            let mut hasher = Sha256::new();
            hasher.update(verifier.as_bytes());
            BASE64.encode(hasher.finalize())
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
// OAuth2 Endpoints
// ============================================================================

/// POST /api/v1/oauth2/register
/// Register a new OAuth2 client application
pub async fn register_client(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RegisterClientRequest>,
) -> Result<Json<ApiResponse<RegisterClientResponse>>, StatusCode> {
    info!("🔐 Registering new OAuth2 client: {}", request.name);

    // Generate client credentials
    let client_id = format!("qnk_client_{}", generate_random_token(16));
    let client_secret = generate_random_token(32);

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
        scopes: vec!["read:balance".to_string(), "send:transaction".to_string()],
        created_at: Utc::now(),
        kyber_public_key,
    };

    state
        .oauth2_storage
        .write()
        .await
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

    // Validate client
    let client = match state
        .oauth2_storage
        .read()
        .await
        .get_client(&params.client_id)
        .await
    {
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
    if !client.redirect_uris.contains(&params.redirect_uri) {
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

    // Redirect to consent screen
    let consent_url = format!(
        "https://quillon.xyz/oauth/consent?client_id={}&redirect_uri={}&scope={}&state={}",
        params.client_id,
        urlencoding::encode(&params.redirect_uri),
        urlencoding::encode(&params.scope.as_deref().unwrap_or("read:balance")),
        urlencoding::encode(&params.state.as_deref().unwrap_or(""))
    );

    debug!("Redirecting to consent screen: {}", consent_url);
    Ok(Redirect::to(&consent_url))
}

/// POST /api/v1/oauth2/consent
/// User grants or denies consent for client access
pub async fn handle_consent(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ConsentRequest>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    info!(
        "🔐 Processing consent from wallet: {} for client: {}",
        request.wallet_address, request.client_id
    );

    if !request.approved {
        info!("❌ User denied consent");
        return Ok(Json(ApiResponse::error("User denied consent".to_string())));
    }

    // Store consent
    let consent = UserConsent {
        wallet_address: request.wallet_address.clone(),
        client_id: request.client_id.clone(),
        scopes: request.scopes.clone(),
        granted_at: Utc::now(),
        expires_at: Some(Utc::now() + Duration::days(365)), // 1 year
    };

    state
        .oauth2_storage
        .write()
        .await
        .store_consent(consent)
        .await;

    // Generate authorization code
    let auth_code = generate_random_token(32);
    let code_record = AuthorizationCode {
        code: auth_code.clone(),
        client_id: request.client_id,
        wallet_address: request.wallet_address,
        redirect_uri: String::new(), // Will be validated during token exchange
        scopes: request.scopes,
        expires_at: Utc::now() + Duration::seconds(AUTH_CODE_EXPIRY_SECONDS),
        code_challenge: None,
        code_challenge_method: None,
    };

    state
        .oauth2_storage
        .write()
        .await
        .store_auth_code(code_record)
        .await;

    info!("✅ Consent granted, authorization code generated");
    Ok(Json(ApiResponse::success(auth_code)))
}

/// POST /api/v1/oauth2/token
/// Exchange authorization code for access token
pub async fn token(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TokenRequest>,
) -> Result<Json<ApiResponse<TokenResponse>>, StatusCode> {
    info!("🔐 OAuth2 token request from client: {}", request.client_id);

    // Validate client credentials
    let client = match state
        .oauth2_storage
        .read()
        .await
        .get_client(&request.client_id)
        .await
    {
        Some(c) if c.client_secret == request.client_secret => c,
        _ => {
            error!("Invalid client credentials");
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
                .write()
                .await
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

            // Generate access token
            let access_token = generate_random_token(32);
            let refresh_token = generate_random_token(32);

            let token_record = AccessToken {
                token: access_token.clone(),
                client_id: auth_code.client_id,
                wallet_address: auth_code.wallet_address,
                scopes: auth_code.scopes.clone(),
                expires_at: Utc::now() + Duration::seconds(TOKEN_EXPIRY_SECONDS),
                refresh_token: Some(refresh_token.clone()),
            };

            state
                .oauth2_storage
                .write()
                .await
                .store_access_token(token_record)
                .await;

            info!("✅ Access token generated");
            Ok(Json(ApiResponse::success(TokenResponse {
                access_token,
                token_type: "Bearer".to_string(),
                expires_in: TOKEN_EXPIRY_SECONDS,
                refresh_token: Some(refresh_token),
                scope: auth_code.scopes.join(" "),
            })))
        }
        "refresh_token" => {
            // Refresh access token
            let refresh_token = request.refresh_token.as_ref().ok_or_else(|| {
                error!("Missing refresh token");
                StatusCode::BAD_REQUEST
            })?;

            // TODO: Implement refresh token logic
            error!("Refresh token not yet implemented");
            Ok(Json(ApiResponse::error(
                "Refresh token not implemented".to_string(),
            )))
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

    // Validate access token
    let access_token = state
        .oauth2_storage
        .read()
        .await
        .get_access_token(token)
        .await
        .ok_or(StatusCode::UNAUTHORIZED)?;

    // Check if expired
    if access_token.expires_at < Utc::now() {
        return Err(StatusCode::UNAUTHORIZED);
    }

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
        if let Ok(address_bytes) = hex::decode(&access_token.wallet_address) {
            if address_bytes.len() == 32 {
                let mut address = [0u8; 32];
                address.copy_from_slice(&address_bytes);
                if let Some(balance) = state.wallet_balances.read().await.get(&address).copied() {
                    user_info["balance"] = serde_json::json!(balance);
                    user_info["balance_qug"] = serde_json::json!(balance as f64 / 100_000_000.0);
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
    Json(request): Json<RevokeRequest>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    info!("🔐 Revoking token");
    state
        .oauth2_storage
        .write()
        .await
        .revoke_token(&request.token)
        .await;
    Ok(Json(ApiResponse::success("Token revoked".to_string())))
}

/// GET /api/v1/oauth2/clients/:client_id
/// Get client information (public data only)
pub async fn get_client_info(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(client_id): axum::extract::Path<String>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    info!("🔍 Client info request for: {}", client_id);

    match state
        .oauth2_storage
        .read()
        .await
        .get_client(&client_id)
        .await
    {
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
