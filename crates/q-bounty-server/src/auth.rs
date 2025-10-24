use axum::{
    async_trait,
    extract::FromRequestParts,
    http::{request::Parts, StatusCode},
    RequestPartsExt,
};
use axum_extra::{
    headers::{authorization::Bearer, Authorization},
    TypedHeader,
};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::fmt::Display;

/// JWT secret key - in production, this should be loaded from environment variable
const JWT_SECRET: &[u8] = b"q-narwhalknight-bounty-jwt-secret-change-in-production";

/// JWT token claims
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    /// Wallet address (hex-encoded)
    pub address: String,
    /// User ID
    pub user_id: String,
    /// Token expiration timestamp (Unix timestamp)
    pub exp: usize,
}

/// Authenticated user information extracted from JWT
#[derive(Debug, Clone)]
pub struct AuthenticatedUser {
    pub address: String,
    pub user_id: String,
}

impl Display for AuthenticatedUser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "User {} ({})", self.user_id, self.address)
    }
}

/// Generate a JWT token for a wallet address
pub fn generate_token(address: &str, user_id: &str) -> Result<String, jsonwebtoken::errors::Error> {
    let expiration = chrono::Utc::now()
        .checked_add_signed(chrono::Duration::hours(24))
        .expect("valid timestamp")
        .timestamp() as usize;

    let claims = Claims {
        address: address.to_string(),
        user_id: user_id.to_string(),
        exp: expiration,
    };

    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(JWT_SECRET),
    )
}

/// Validate a JWT token and extract claims
pub fn validate_token(token: &str) -> Result<Claims, jsonwebtoken::errors::Error> {
    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(JWT_SECRET),
        &Validation::default(),
    )?;

    Ok(token_data.claims)
}

/// Axum extractor for authenticated requests
#[async_trait]
impl<S> FromRequestParts<S> for AuthenticatedUser
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, String);

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Extract the authorization header
        let TypedHeader(Authorization(bearer)) = parts
            .extract::<TypedHeader<Authorization<Bearer>>>()
            .await
            .map_err(|_| {
                (
                    StatusCode::UNAUTHORIZED,
                    "Missing or invalid Authorization header".to_string(),
                )
            })?;

        // Validate the token
        let claims = validate_token(bearer.token()).map_err(|e| {
            (
                StatusCode::UNAUTHORIZED,
                format!("Invalid token: {}", e),
            )
        })?;

        Ok(AuthenticatedUser {
            address: claims.address,
            user_id: claims.user_id,
        })
    }
}

/// Verify an Ed25519 signature
pub fn verify_signature(
    address_bytes: &[u8; 32],
    message: &[u8],
    signature: &[u8; 64],
) -> Result<(), String> {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};

    let public_key = VerifyingKey::from_bytes(address_bytes)
        .map_err(|e| format!("Invalid public key: {}", e))?;

    let sig = Signature::from_bytes(signature);

    public_key
        .verify(message, &sig)
        .map_err(|e| format!("Signature verification failed: {}", e))
}
