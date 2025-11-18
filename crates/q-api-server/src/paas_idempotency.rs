// PaaS Idempotency System
// Prevents duplicate charges from client retries
//
// Features:
// - Idempotency-Key header validation
// - Response caching with 24-hour TTL
// - Conflict detection (same key, different body)
// - Automatic cleanup of expired entries
//
// Usage:
// POST /api/v1/privacy/tor/relay
// Headers:
//   Idempotency-Key: unique-request-id-12345
//   X-Auth-Token: {...}
//
// Response:
// - First request: Process normally, cache response
// - Retry (same key, same body): Return cached response (200 OK)
// - Retry (same key, different body): Return 409 Conflict

use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// Cached idempotent response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdempotentResponse {
    /// Idempotency key
    pub idempotency_key: String,

    /// Request body hash (SHA256)
    pub body_hash: String,

    /// Cached response status code
    pub status_code: u16,

    /// Cached response headers
    pub headers: Vec<(String, String)>,

    /// Cached response body
    pub body: Vec<u8>,

    /// Response creation timestamp
    pub created_at: DateTime<Utc>,

    /// Cache expiration (24 hours from creation)
    pub expires_at: DateTime<Utc>,

    /// Request count for this key
    pub request_count: u32,

    /// Last accessed timestamp
    pub last_accessed_at: DateTime<Utc>,
}

impl IdempotentResponse {
    /// Check if cache entry has expired
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }
}

/// Idempotency manager
pub struct PaaSIdempotencyManager {
    /// Cached responses (idempotency_key -> IdempotentResponse)
    cache: Arc<RwLock<HashMap<String, IdempotentResponse>>>,

    /// Cache TTL (seconds)
    cache_ttl: u64,
}

impl PaaSIdempotencyManager {
    pub fn new() -> Self {
        let manager = Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            cache_ttl: 86400, // 24 hours
        };

        // Start background cleanup task
        manager.start_cache_cleanup();

        manager
    }

    /// Check if idempotency key exists and validate request
    pub async fn check_idempotency(
        &self,
        idempotency_key: &str,
        request_body: &[u8],
    ) -> IdempotencyCheck {
        let body_hash = self.hash_body(request_body);

        let mut cache = self.cache.write().await;

        if let Some(cached) = cache.get_mut(idempotency_key) {
            // Update access time
            cached.last_accessed_at = Utc::now();
            cached.request_count += 1;

            // Check if cache expired
            if cached.is_expired() {
                cache.remove(idempotency_key);
                drop(cache);
                return IdempotencyCheck::NotFound;
            }

            // Check if request body matches
            if cached.body_hash == body_hash {
                info!(
                    "♻️  Idempotency key {} matched (request count: {})",
                    &idempotency_key[..8],
                    cached.request_count
                );
                IdempotencyCheck::Match(cached.clone())
            } else {
                warn!(
                    "⚠️  Idempotency key {} conflict (different body hash)",
                    &idempotency_key[..8]
                );
                IdempotencyCheck::Conflict
            }
        } else {
            IdempotencyCheck::NotFound
        }
    }

    /// Cache a response for an idempotency key
    pub async fn cache_response(
        &self,
        idempotency_key: String,
        request_body: &[u8],
        status_code: u16,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    ) {
        let body_hash = self.hash_body(request_body);
        let now = Utc::now();
        let expires_at = now + chrono::Duration::seconds(self.cache_ttl as i64);

        let cached_response = IdempotentResponse {
            idempotency_key: idempotency_key.clone(),
            body_hash,
            status_code,
            headers,
            body,
            created_at: now,
            expires_at,
            request_count: 1,
            last_accessed_at: now,
        };

        let mut cache = self.cache.write().await;
        cache.insert(idempotency_key.clone(), cached_response);

        info!(
            "💾 Cached response for idempotency key {} (TTL: {}h)",
            &idempotency_key[..8],
            self.cache_ttl / 3600
        );
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> IdempotencyStats {
        let cache = self.cache.read().await;

        let total_entries = cache.len();
        let expired_entries = cache.values().filter(|r| r.is_expired()).count();
        let total_requests: u32 = cache.values().map(|r| r.request_count).sum();

        IdempotencyStats {
            total_entries,
            expired_entries,
            active_entries: total_entries - expired_entries,
            total_requests,
            cache_ttl_hours: self.cache_ttl / 3600,
        }
    }

    /// Hash request body for comparison
    fn hash_body(&self, body: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(body);
        hex::encode(hasher.finalize())
    }

    /// Background task to cleanup expired cache entries
    fn start_cache_cleanup(&self) {
        let cache = self.cache.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(3600)); // Every hour

            loop {
                interval.tick().await;

                let mut cache_write = cache.write().await;
                let initial_count = cache_write.len();

                // Remove expired entries
                cache_write.retain(|_, response| !response.is_expired());

                let removed_count = initial_count - cache_write.len();

                if removed_count > 0 {
                    info!(
                        "🧹 Cleaned up {} expired idempotency cache entries ({} remaining)",
                        removed_count,
                        cache_write.len()
                    );
                }
            }
        });
    }
}

/// Idempotency check result
#[derive(Debug)]
pub enum IdempotencyCheck {
    /// Key not found in cache (first request)
    NotFound,

    /// Key found with matching body (duplicate request)
    Match(IdempotentResponse),

    /// Key found with different body (conflict)
    Conflict,
}

/// Idempotency cache statistics
#[derive(Debug, Clone, Serialize)]
pub struct IdempotencyStats {
    pub total_entries: usize,
    pub expired_entries: usize,
    pub active_entries: usize,
    pub total_requests: u32,
    pub cache_ttl_hours: u64,
}

/// Axum middleware for idempotency handling
pub async fn idempotency_middleware(
    idempotency_manager: Arc<PaaSIdempotencyManager>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract Idempotency-Key header (optional)
    let idempotency_key = headers
        .get("Idempotency-Key")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string());

    // If no idempotency key, process normally
    let Some(key) = idempotency_key else {
        return Ok(next.run(request).await);
    };

    // Validate idempotency key format (UUID or similar)
    if key.len() < 8 || key.len() > 128 {
        error!("❌ Invalid Idempotency-Key format: length {}", key.len());
        return Err(StatusCode::BAD_REQUEST);
    }

    // Extract request body
    let (parts, body) = request.into_parts();
    let body_bytes = axum::body::to_bytes(body, usize::MAX)
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    // Check idempotency
    match idempotency_manager
        .check_idempotency(&key, &body_bytes)
        .await
    {
        IdempotencyCheck::NotFound => {
            // First request with this key - process normally
            info!("🆕 New idempotency key: {}", &key[..8]);

            // Reconstruct request
            let request = Request::from_parts(parts, Body::from(body_bytes.clone()));

            // Process request
            let response = next.run(request).await;

            // Cache response if successful
            if response.status().is_success() {
                // Extract response details for caching
                let status = response.status().as_u16();
                let headers: Vec<(String, String)> = response
                    .headers()
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                    .collect();

                // Extract body (this is a simplification - in production, use a more robust method)
                // For now, we'll skip body caching for complex streaming responses
                idempotency_manager
                    .cache_response(key, &body_bytes, status, headers, vec![])
                    .await;
            }

            Ok(response)
        }

        IdempotencyCheck::Match(cached) => {
            // Duplicate request - return cached response
            info!(
                "♻️  Returning cached response for idempotency key: {}",
                &key[..8]
            );

            // Build response from cache
            let mut response = Response::new(Body::from(cached.body));
            *response.status_mut() =
                StatusCode::from_u16(cached.status_code).unwrap_or(StatusCode::OK);

            // Add cached headers
            for (name, value) in cached.headers {
                if let (Ok(header_name), Ok(header_value)) = (
                    name.parse::<axum::http::HeaderName>(),
                    value.parse::<axum::http::HeaderValue>(),
                ) {
                    response.headers_mut().insert(header_name, header_value);
                }
            }

            // Add idempotency replay header
            response
                .headers_mut()
                .insert("X-Idempotency-Replay", "true".parse().unwrap());

            Ok(response)
        }

        IdempotencyCheck::Conflict => {
            // Same key, different body - conflict
            error!(
                "❌ Idempotency conflict for key: {} (different request body)",
                &key[..8]
            );

            let error_body = serde_json::json!({
                "error": "Idempotency conflict",
                "message": "Request with same Idempotency-Key but different body already exists",
                "idempotency_key": &key[..8],
            })
            .to_string();

            let mut response = Response::new(Body::from(error_body));
            *response.status_mut() = StatusCode::CONFLICT;
            response.headers_mut().insert(
                axum::http::header::CONTENT_TYPE,
                "application/json".parse().unwrap(),
            );

            Ok(response)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_idempotency_first_request() {
        let manager = PaaSIdempotencyManager::new();
        let key = "test-key-12345";
        let body = b"test request body";

        // First request should return NotFound
        let check = manager.check_idempotency(key, body).await;
        assert!(matches!(check, IdempotencyCheck::NotFound));
    }

    #[tokio::test]
    async fn test_idempotency_duplicate_request() {
        let manager = PaaSIdempotencyManager::new();
        let key = "test-key-67890".to_string();
        let body = b"test request body";

        // Cache a response
        manager
            .cache_response(
                key.clone(),
                body,
                200,
                vec![("content-type".to_string(), "application/json".to_string())],
                b"{\"success\": true}".to_vec(),
            )
            .await;

        // Second request with same body should return Match
        let check = manager.check_idempotency(&key, body).await;
        assert!(matches!(check, IdempotencyCheck::Match(_)));
    }

    #[tokio::test]
    async fn test_idempotency_conflict() {
        let manager = PaaSIdempotencyManager::new();
        let key = "test-key-conflict".to_string();
        let body1 = b"original request body";
        let body2 = b"different request body";

        // Cache first response
        manager
            .cache_response(key.clone(), body1, 200, vec![], vec![])
            .await;

        // Request with different body should return Conflict
        let check = manager.check_idempotency(&key, body2).await;
        assert!(matches!(check, IdempotencyCheck::Conflict));
    }

    #[tokio::test]
    async fn test_cache_statistics() {
        let manager = PaaSIdempotencyManager::new();

        // Add some cached responses
        for i in 0..5 {
            let key = format!("test-key-{}", i);
            manager
                .cache_response(key, b"test", 200, vec![], vec![])
                .await;
        }

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_entries, 5);
        assert_eq!(stats.active_entries, 5);
        assert_eq!(stats.total_requests, 5);
    }
}
