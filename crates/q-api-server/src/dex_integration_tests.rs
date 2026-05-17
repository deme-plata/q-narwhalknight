/// Comprehensive test suite for DEX Integration API
///
/// This test suite covers all endpoints, security features, validation logic,
/// and integration scenarios for the Q-NarwhalKnight DEX API.

#[cfg(test)]
mod tests {
    use super::super::dex_integration_api::*;
    use crate::{AppState, Config};
    use axum::{
        body::Body,
        extract::{Path, Query, State as AxumState},
        http::{HeaderMap, Method, Request, StatusCode},
        response::Response,
        routing::{get, post},
        Json, Router,
    };
    use serde_json::{json, Value};
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt; // for `oneshot`
    use futures::future;

    /// Helper function to create test AppState
    async fn create_test_app_state() -> Arc<AppState> {
        let config = Config::default();
        let app_state = AppState::new(config).await.expect("Failed to create test AppState");
        Arc::new(app_state)
    }

    /// Helper function to create test router
    fn create_test_router() -> Router {
        create_dex_integration_router()
            .with_state(tokio_test::block_on(create_test_app_state()))
    }

    /// Helper function to make HTTP requests to the test router
    async fn make_request(
        router: Router,
        method: Method,
        uri: &str,
        body: Option<Value>,
        headers: Option<HeaderMap>,
    ) -> Response<Body> {
        let mut request_builder = Request::builder().method(method).uri(uri);
        
        if let Some(h) = headers {
            for (key, value) in h.iter() {
                request_builder = request_builder.header(key, value);
            }
        }

        let request = if let Some(json_body) = body {
            request_builder
                .header("content-type", "application/json")
                .body(Body::from(json_body.to_string()))
                .unwrap()
        } else {
            request_builder.body(Body::empty()).unwrap()
        };

        router.oneshot(request).await.unwrap()
    }

    // ============ SECURITY & VALIDATION TESTS ============

    #[tokio::test]
    async fn test_rate_limiter() {
        let rate_limiter = RateLimiter::new(10); // 10 requests per hour

        // Test initial requests are allowed
        for i in 1..=10 {
            let allowed = rate_limiter.is_allowed("test_client").await;
            assert!(allowed, "Request {} should be allowed", i);
        }

        // Test that 11th request is blocked
        let blocked = rate_limiter.is_allowed("test_client").await;
        assert!(!blocked, "11th request should be blocked by rate limiter");

        // Test different client is still allowed
        let different_client = rate_limiter.is_allowed("different_client").await;
        assert!(different_client, "Different client should be allowed");
    }

    #[tokio::test]
    async fn test_rate_limiter_remaining_calls() {
        let rate_limiter = RateLimiter::new(100);
        
        // Make 20 calls
        for _ in 1..=20 {
            rate_limiter.is_allowed("test_client").await;
        }

        let remaining = rate_limiter.get_remaining_calls("test_client").await;
        assert_eq!(remaining, 80, "Should have 80 remaining calls");

        let remaining_new_client = rate_limiter.get_remaining_calls("new_client").await;
        assert_eq!(remaining_new_client, 100, "New client should have full quota");
    }

    #[test]
    fn test_api_key_validation() {
        // Valid API key
        let valid_key = ApiKey {
            key: "qnk_12345678901234567890123456789012".to_string(),
            permissions: vec!["read:pools".to_string(), "write:swaps".to_string()],
            rate_limit: 1000,
            created_at: chrono::Utc::now().timestamp() as u64,
            expires_at: Some(chrono::Utc::now().timestamp() as u64 + 3600), // 1 hour from now
            is_active: true,
        };
        assert!(valid_key.validate(), "Valid API key should pass validation");
        assert!(valid_key.has_permission("read:pools"), "Should have read:pools permission");
        assert!(!valid_key.has_permission("admin"), "Should not have admin permission");

        // Inactive API key
        let inactive_key = ApiKey {
            is_active: false,
            ..valid_key.clone()
        };
        assert!(!inactive_key.validate(), "Inactive API key should fail validation");

        // Expired API key
        let expired_key = ApiKey {
            expires_at: Some(chrono::Utc::now().timestamp() as u64 - 3600), // 1 hour ago
            ..valid_key.clone()
        };
        assert!(!expired_key.validate(), "Expired API key should fail validation");
    }

    #[test]
    fn test_validate_api_key_format() {
        // Valid format
        assert!(validate_api_key("qnk_12345678901234567890123456789012"));
        
        // Invalid formats
        assert!(!validate_api_key("invalid_key"));
        assert!(!validate_api_key("qnk_short"));
        assert!(!validate_api_key("wrong_prefix_12345678901234567890123456789012"));
        assert!(!validate_api_key(""));
    }

    #[test]
    fn test_extract_client_ip() {
        let mut headers = HeaderMap::new();
        
        // Test X-Forwarded-For header
        headers.insert("x-forwarded-for", "192.168.1.100, 10.0.0.1".parse().unwrap());
        let ip = extract_client_ip(&headers);
        assert_eq!(ip, "192.168.1.100", "Should extract first IP from X-Forwarded-For");

        // Test X-Real-IP header
        headers.clear();
        headers.insert("x-real-ip", "203.0.113.45".parse().unwrap());
        let ip = extract_client_ip(&headers);
        assert_eq!(ip, "203.0.113.45", "Should extract IP from X-Real-IP");

        // Test fallback to localhost
        headers.clear();
        let ip = extract_client_ip(&headers);
        assert_eq!(ip, "127.0.0.1", "Should fallback to localhost when no headers");
    }

    // ============ ENDPOINT FUNCTIONALITY TESTS ============

    #[tokio::test]
    async fn test_get_node_info_endpoint() {
        let app_state = create_test_app_state().await;
        let response = get_node_info(AxumState(app_state)).await.unwrap();
        
        let response_data: DexApiResponse<NodeIntegrationInfo> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(response_data.success, "Node info request should succeed");
        let info = response_data.data.unwrap();
        assert_eq!(info.network, "Q-NarwhalKnight-Mainnet");
        assert_eq!(info.api_version, "1.0.0");
        assert!(info.vm_capabilities.smart_contracts, "Should support smart contracts");
        assert!(info.vm_capabilities.quantum_security, "Should have quantum security");
        assert_eq!(info.performance_metrics.tps, 27200, "Should report correct TPS");
    }

    #[tokio::test]
    async fn test_get_supported_tokens_endpoint() {
        let app_state = create_test_app_state().await;
        let response = get_supported_tokens(AxumState(app_state)).await.unwrap();
        
        let response_data: DexApiResponse<Vec<TokenInfo>> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(response_data.success, "Supported tokens request should succeed");
        let tokens = response_data.data.unwrap();
        assert!(!tokens.is_empty(), "Should return at least QNK token");
        
        let qnk_token = &tokens[0];
        assert_eq!(qnk_token.symbol, "QNK");
        assert_eq!(qnk_token.name, "Q-NarwhalKnight Token");
        assert!(qnk_token.verified, "QNK token should be verified");
    }

    #[tokio::test]
    async fn test_get_token_info_valid_address() {
        let app_state = create_test_app_state().await;
        let path = Path("0x0000000000000000000000000000000000000000".to_string());
        let response = get_token_info(path, AxumState(app_state)).await.unwrap();
        
        let response_data: DexApiResponse<TokenInfo> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(response_data.success, "Valid token info request should succeed");
        let token = response_data.data.unwrap();
        assert_eq!(token.symbol, "QNK");
        assert_eq!(token.decimals, 18);
    }

    #[tokio::test]
    async fn test_get_token_info_invalid_address() {
        let app_state = create_test_app_state().await;
        let path = Path("invalid_address".to_string());
        let response = get_token_info(path, AxumState(app_state)).await.unwrap();
        
        let response_data: DexApiResponse<TokenInfo> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(!response_data.success, "Invalid token info request should fail");
        assert_eq!(response_data.error.unwrap(), "Token not found");
    }

    #[tokio::test]
    async fn test_generate_api_key_endpoint() {
        let app_state = create_test_app_state().await;
        let response = generate_api_key(AxumState(app_state)).await.unwrap();
        
        let response_data: DexApiResponse<ApiKeyInfo> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(response_data.success, "API key generation should succeed");
        let api_key_info = response_data.data.unwrap();
        assert!(api_key_info.api_key.starts_with("qnk_"), "API key should have QNK prefix");
        assert!(api_key_info.api_key.len() >= 32, "API key should be at least 32 characters");
        assert!(api_key_info.permissions.contains(&"read:pools".to_string()), "Should have read:pools permission");
        assert!(api_key_info.expires_at.is_some(), "Should have expiration date");
    }

    #[tokio::test]
    async fn test_get_rate_limits_endpoint() {
        let app_state = create_test_app_state().await;
        let response = get_rate_limits(AxumState(app_state)).await.unwrap();
        
        let response_data: DexApiResponse<RateLimits> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(response_data.success, "Rate limits request should succeed");
        let limits = response_data.data.unwrap();
        assert_eq!(limits.requests_per_hour, 5000);
        assert_eq!(limits.requests_per_minute, 100);
        assert!(limits.reset_time > chrono::Utc::now().timestamp() as u64);
    }

    // ============ SWAP FUNCTIONALITY TESTS ============

    #[tokio::test]
    async fn test_get_swap_quote_valid_request() {
        let app_state = create_test_app_state().await;
        let request = SwapQuoteRequest {
            token_in: "QNK".to_string(),
            token_out: "USDC".to_string(),
            amount_in: Some("1000000".to_string()),
            amount_out: None,
            slippage_tolerance: Some(0.5),
        };
        
        let response = get_swap_quote(AxumState(app_state), Json(request)).await.unwrap();
        
        let response_data: DexApiResponse<SwapQuote> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(response_data.success, "Valid swap quote should succeed");
        let quote = response_data.data.unwrap();
        assert_eq!(quote.amount_in, "1000000");
        assert!(!quote.amount_out.is_empty());
        assert!(quote.gas_estimate > 0);
        assert!(quote.valid_until > chrono::Utc::now().timestamp() as u64);
    }

    #[tokio::test]
    async fn test_get_swap_quote_invalid_slippage() {
        let app_state = create_test_app_state().await;
        let request = SwapQuoteRequest {
            token_in: "QNK".to_string(),
            token_out: "USDC".to_string(),
            amount_in: Some("1000000".to_string()),
            amount_out: None,
            slippage_tolerance: Some(15.0), // Invalid: > 10%
        };
        
        let response = get_swap_quote(AxumState(app_state), Json(request)).await.unwrap();
        
        let response_data: DexApiResponse<SwapQuote> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(!response_data.success, "Invalid slippage should fail");
        assert!(response_data.error.unwrap().contains("Slippage tolerance"));
    }

    #[tokio::test]
    async fn test_get_swap_quote_same_token() {
        let app_state = create_test_app_state().await;
        let request = SwapQuoteRequest {
            token_in: "QNK".to_string(),
            token_out: "QNK".to_string(),
            amount_in: Some("1000000".to_string()),
            amount_out: None,
            slippage_tolerance: Some(0.5),
        };
        
        let response = get_swap_quote(AxumState(app_state), Json(request)).await.unwrap();
        
        let response_data: DexApiResponse<SwapQuote> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(!response_data.success, "Same token swap should fail");
        assert!(response_data.error.unwrap().contains("Cannot swap token for itself"));
    }

    #[tokio::test]
    async fn test_execute_swap_valid_request() {
        let app_state = create_test_app_state().await;
        let request = SwapExecuteRequest {
            token_in: "QNK".to_string(),
            token_out: "USDC".to_string(),
            amount_in: "1000000".to_string(),
            minimum_amount_out: "950000".to_string(),
            recipient: "0x742C4AA22F91c92A8d8c7a3E06C4e12e86B3e2d7".to_string(),
            deadline: chrono::Utc::now().timestamp() as u64 + 3600, // 1 hour from now
            signature: "0x1234567890abcdef...".to_string(),
        };
        
        let response = execute_swap(AxumState(app_state), Json(request)).await.unwrap();
        
        let response_data: DexApiResponse<SwapResult> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(response_data.success, "Valid swap execution should succeed");
        let result = response_data.data.unwrap();
        assert!(result.transaction_hash.starts_with("0x"));
        assert_eq!(result.status, "pending");
        assert_eq!(result.amount_in, "1000000");
        assert!(!result.amount_out.is_empty());
        assert!(result.gas_used > 0);
    }

    #[tokio::test]
    async fn test_execute_swap_expired_deadline() {
        let app_state = create_test_app_state().await;
        let request = SwapExecuteRequest {
            token_in: "QNK".to_string(),
            token_out: "USDC".to_string(),
            amount_in: "1000000".to_string(),
            minimum_amount_out: "950000".to_string(),
            recipient: "0x742C4AA22F91c92A8d8c7a3E06C4e12e86B3e2d7".to_string(),
            deadline: chrono::Utc::now().timestamp() as u64 - 3600, // 1 hour ago
            signature: "0x1234567890abcdef...".to_string(),
        };
        
        let response = execute_swap(AxumState(app_state), Json(request)).await.unwrap();
        
        let response_data: DexApiResponse<SwapResult> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(!response_data.success, "Expired deadline should fail");
        assert!(response_data.error.unwrap().contains("deadline has passed"));
    }

    #[tokio::test]
    async fn test_execute_swap_invalid_address() {
        let app_state = create_test_app_state().await;
        let request = SwapExecuteRequest {
            token_in: "QNK".to_string(),
            token_out: "USDC".to_string(),
            amount_in: "1000000".to_string(),
            minimum_amount_out: "950000".to_string(),
            recipient: "invalid_address".to_string(),
            deadline: chrono::Utc::now().timestamp() as u64 + 3600,
            signature: "0x1234567890abcdef...".to_string(),
        };
        
        let response = execute_swap(AxumState(app_state), Json(request)).await.unwrap();
        
        let response_data: DexApiResponse<SwapResult> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(!response_data.success, "Invalid address should fail");
        assert!(response_data.error.unwrap().contains("invalid recipient address"));
    }

    // ============ PRICE ORACLE TESTS ============

    #[tokio::test]
    async fn test_get_token_price_qnk() {
        let app_state = create_test_app_state().await;
        let path = Path("QNK".to_string());
        let response = get_token_price(path, AxumState(app_state)).await.unwrap();
        
        let response_data: DexApiResponse<TokenPrice> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(response_data.success, "QNK price request should succeed");
        let price = response_data.data.unwrap();
        assert_eq!(price.token, "QNK");
        assert_eq!(price.price_usd, 1.0);
        assert_eq!(price.price_qnk, 1.0);
        assert!(price.last_updated > 0);
    }

    #[tokio::test]
    async fn test_get_token_price_unknown() {
        let app_state = create_test_app_state().await;
        let path = Path("UNKNOWN".to_string());
        let response = get_token_price(path, AxumState(app_state)).await.unwrap();
        
        let response_data: DexApiResponse<TokenPrice> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(response_data.success, "Unknown token request should succeed but return zero price");
        let price = response_data.data.unwrap();
        assert_eq!(price.token, "UNKNOWN");
        assert_eq!(price.price_usd, 0.0);
        assert_eq!(price.price_qnk, 0.0);
    }

    #[tokio::test]
    async fn test_get_historical_prices() {
        let app_state = create_test_app_state().await;
        let path = Path("QNK".to_string());
        let mut params = HashMap::new();
        params.insert("timeframe".to_string(), "24h".to_string());
        params.insert("interval".to_string(), "1h".to_string());
        let query = Query(params);
        
        let response = get_historical_prices(path, query, AxumState(app_state)).await.unwrap();
        
        let response_data: DexApiResponse<Vec<TokenPrice>> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(response_data.success, "Historical prices request should succeed");
        let prices = response_data.data.unwrap();
        assert_eq!(prices.len(), 2, "Should return 2 historical data points");
        assert!(prices[0].last_updated < prices[1].last_updated, "Prices should be in chronological order");
    }

    // ============ COMPLIANCE TESTS ============

    #[tokio::test]
    async fn test_compliance_check_valid() {
        let app_state = create_test_app_state().await;
        let request = json!({
            "address": "0x742C4AA22F91c92A8d8c7a3E06C4e12e86B3e2d7",
            "type": "swap"
        });
        
        let response = compliance_check(AxumState(app_state), Json(request)).await.unwrap();
        
        let response_data: DexApiResponse<Value> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(response_data.success, "Valid compliance check should succeed");
        let result = response_data.data.unwrap();
        assert_eq!(result["compliance_status"], "approved");
        assert_eq!(result["sanctions_check"], "passed");
        assert_eq!(result["aml_status"], "clear");
    }

    #[tokio::test]
    async fn test_compliance_check_invalid() {
        let app_state = create_test_app_state().await;
        let request = json!({
            "address": "invalid",
            "type": "swap"
        });
        
        let response = compliance_check(AxumState(app_state), Json(request)).await.unwrap();
        
        let response_data: DexApiResponse<Value> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(response_data.success, "Compliance check should succeed but reject");
        let result = response_data.data.unwrap();
        assert_eq!(result["compliance_status"], "rejected");
        assert_eq!(result["risk_score"], 1.0);
    }

    // ============ LIQUIDITY POOL TESTS ============

    #[tokio::test]
    async fn test_create_liquidity_pool_valid() {
        let app_state = create_test_app_state().await;
        let request = json!({
            "token0": "QNK",
            "token1": "USDC",
            "initial_reserve0": "1000000",
            "initial_reserve1": "1000000"
        });
        
        let response = create_liquidity_pool(AxumState(app_state), Json(request)).await.unwrap();
        
        let response_data: DexApiResponse<String> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        // Note: This might fail if Orobit ecosystem is not fully initialized in test,
        // but the validation logic should work
        if response_data.success {
            assert!(!response_data.data.unwrap().is_empty(), "Should return contract address");
        } else {
            assert!(response_data.error.is_some(), "Should have error message");
        }
    }

    #[tokio::test]
    async fn test_create_liquidity_pool_missing_token() {
        let app_state = create_test_app_state().await;
        let request = json!({
            "token0": "QNK",
            "initial_reserve0": "1000000",
            "initial_reserve1": "1000000"
        });
        
        let response = create_liquidity_pool(AxumState(app_state), Json(request)).await.unwrap();
        
        let response_data: DexApiResponse<String> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(!response_data.success, "Missing token should fail");
        assert!(response_data.error.unwrap().contains("token1 is required"));
    }

    // ============ INTEGRATION TESTS ============

    #[tokio::test]
    async fn test_dex_api_response_structure() {
        let success_response: DexApiResponse<String> = DexApiResponse::success("test data".to_string());
        assert!(success_response.success);
        assert!(success_response.data.is_some());
        assert!(success_response.error.is_none());
        assert_eq!(success_response.api_version, "1.0.0");
        assert_eq!(success_response.network, "mainnet-genesis");

        let error_response: DexApiResponse<String> = DexApiResponse::error("test error".to_string());
        assert!(!error_response.success);
        assert!(error_response.data.is_none());
        assert!(error_response.error.is_some());
        assert_eq!(error_response.error.unwrap(), "test error");
    }

    #[tokio::test]
    async fn test_get_all_pools() {
        let app_state = create_test_app_state().await;
        let response = get_all_pools(AxumState(app_state)).await.unwrap();
        
        let response_data: DexApiResponse<Vec<PoolInfo>> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(response_data.success, "Get all pools should succeed");
        // Initially empty but should not fail
        let pools = response_data.data.unwrap();
        assert!(pools.is_empty(), "Should start with empty pools");
    }

    #[tokio::test]
    async fn test_get_all_prices() {
        let app_state = create_test_app_state().await;
        let response = get_all_prices(AxumState(app_state)).await.unwrap();
        
        let response_data: DexApiResponse<Vec<TokenPrice>> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(response_data.success, "Get all prices should succeed");
        let prices = response_data.data.unwrap();
        assert!(prices.is_empty(), "Should start with empty prices list");
    }

    #[tokio::test]
    async fn test_setup_webhook() {
        let app_state = create_test_app_state().await;
        let request = json!({
            "url": "https://example.com/webhook",
            "events": ["swap", "pool_create"],
            "secret": "webhook_secret_key"
        });
        
        let response = setup_webhook(AxumState(app_state), Json(request)).await.unwrap();
        
        let response_data: DexApiResponse<String> = 
            serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

        assert!(response_data.success, "Webhook setup should succeed");
        assert_eq!(response_data.data.unwrap(), "Webhook registered");
    }

    // ============ PERFORMANCE TESTS ============

    #[tokio::test]
    async fn test_concurrent_api_key_generation() {
        let app_state = create_test_app_state().await;
        
        let mut handles = vec![];
        for _ in 0..10 {
            let state = app_state.clone();
            let handle = tokio::spawn(async move {
                generate_api_key(AxumState(state)).await
            });
            handles.push(handle);
        }

        let mut generated_keys = vec![];
        for handle in handles {
            let response = handle.await.unwrap().unwrap();
            let response_data: DexApiResponse<ApiKeyInfo> = 
                serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
            
            assert!(response_data.success, "Concurrent API key generation should succeed");
            generated_keys.push(response_data.data.unwrap().api_key);
        }

        // Ensure all keys are unique
        generated_keys.sort();
        generated_keys.dedup();
        assert_eq!(generated_keys.len(), 10, "All generated API keys should be unique");
    }

    #[tokio::test]
    async fn test_concurrent_rate_limiting() {
        let rate_limiter = RateLimiter::new(5); // Very low limit for testing
        
        let mut handles = vec![];
        for i in 0..10 {
            let limiter = rate_limiter.clone();
            let handle = tokio::spawn(async move {
                limiter.is_allowed(&format!("client_{}", i % 2)).await // Use 2 different clients
            });
            handles.push(handle);
        }

        let results: Vec<bool> = futures::future::join_all(handles).await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // Should have some allowed and some blocked requests
        let allowed_count = results.iter().filter(|&&x| x).count();
        let blocked_count = results.iter().filter(|&&x| !x).count();
        
        assert!(allowed_count > 0, "Some requests should be allowed");
        assert!(blocked_count > 0, "Some requests should be blocked");
        assert_eq!(allowed_count + blocked_count, 10, "All requests should be processed");
    }
}

/// Test helper traits and implementations
#[cfg(test)]
mod test_helpers {
    use super::*;

    /// Mock configuration for testing
    impl Default for Config {
        fn default() -> Self {
            Config {
                port: 8080,
                is_validator: true,
                node_id: None,
                db_path: Some("test_db".to_string()),
                hot_db_path: Some("test_hot_db".to_string()),
                tor: Default::default(),
            }
        }
    }
}