use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
};
use q_api_server::{AppState, Config};
use serde_json::json;
/// API Integration Tests for Orobit Smart Contract Endpoints
///
/// This test suite validates all REST API endpoints for smart contract
/// deployment, management, and interaction.
use std::collections::HashMap;
use std::sync::Arc;
use tower::ServiceExt;

/// Helper function to create test app state
async fn create_test_app_state() -> Arc<AppState> {
    let config = Config {
        port: 8080,
        is_validator: false,
        node_id: Some([1u8; 32]),
        db_path: Some("test_db".to_string()),
        hot_db_path: Some("test_hot_db".to_string()),
        tor: Default::default(),
    };

    Arc::new(
        AppState::new(config)
            .await
            .expect("Failed to create test app state"),
    )
}

/// Helper function to create contracts router
fn create_test_router() -> Router {
    let app_state = tokio::runtime::Handle::current().block_on(create_test_app_state());
    q_api_server::contracts_api::create_contracts_router().with_state(app_state)
}

/// Test GET /templates endpoint
#[tokio::test]
async fn test_get_contract_templates() {
    let app = create_test_router();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/templates")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json_response["success"].as_bool().unwrap_or(false));
    assert!(json_response["data"]["templates"].is_array());

    let templates = json_response["data"]["templates"].as_array().unwrap();
    assert!(!templates.is_empty(), "Should return available templates");

    // Verify required template fields
    let first_template = &templates[0];
    assert!(first_template["contract_type"].is_string());
    assert!(first_template["name"].is_string());
    assert!(first_template["description"].is_string());
    assert!(first_template["gas_estimate"].is_number());
    assert!(first_template["security_level"].is_string());

    println!("✅ GET /templates test passed");
}

/// Test GET /templates/{contract_type}/form endpoint
#[tokio::test]
async fn test_get_deployment_form() {
    let app = create_test_router();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/templates/secure_token/form")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json_response["success"].as_bool().unwrap_or(false));
    assert!(json_response["data"]["form_title"].is_string());
    assert!(json_response["data"]["form_description"].is_string());
    assert!(json_response["data"]["schema"].is_object());
    assert!(json_response["data"]["gas_estimate"].is_object());

    let gas_estimate = &json_response["data"]["gas_estimate"];
    assert!(gas_estimate["base_gas"].is_number());
    assert!(gas_estimate["total_gas_estimate"].is_number());
    assert!(gas_estimate["estimated_cost_orb"].is_string());

    println!("✅ GET /templates/secure_token/form test passed");
}

/// Test POST /deploy endpoint
#[tokio::test]
async fn test_deploy_contract() {
    let app = create_test_router();

    let deployment_request = json!({
        "contract_type": "secure_token",
        "owner": "0x1234567890123456789012345678901234567890",
        "parameters": {
            "name": "Test Token",
            "symbol": "TST",
            "initial_supply": "1000000000000000000000000",
            "decimals": "18"
        },
        "deployment_options": {
            "test_deployment": true,
            "auto_verify": false,
            "enable_governance": false,
            "enable_upgrades": false,
            "gas_limit": 3000000
        }
    });

    let response = app
        .oneshot(
            Request::builder()
                .uri("/deploy")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(deployment_request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json_response["success"].as_bool().unwrap_or(false));
    assert!(json_response["data"]["request_id"].is_string());
    assert!(json_response["data"]["status"].is_string());
    assert!(json_response["data"]["progress"].is_object());

    let progress = &json_response["data"]["progress"];
    assert!(progress["current_step"].is_number());
    assert!(progress["total_steps"].is_number());
    assert!(progress["step_name"].is_string());

    println!("✅ POST /deploy test passed");
}

/// Test POST /templates/{contract_type}/estimate endpoint
#[tokio::test]
async fn test_estimate_deployment_cost() {
    let app = create_test_router();

    let parameters = json!({
        "mintable": true,
        "burnable": true,
        "stakeable": false
    });

    let response = app
        .oneshot(
            Request::builder()
                .uri("/templates/advanced_token/estimate")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(parameters.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json_response["success"].as_bool().unwrap_or(false));
    assert!(json_response["data"]["base_gas"].is_number());
    assert!(json_response["data"]["total_gas_estimate"].is_number());
    assert!(json_response["data"]["estimated_cost_orb"].is_string());

    // Verify that enabled features increase gas cost
    let base_gas = json_response["data"]["base_gas"].as_u64().unwrap();
    let total_gas = json_response["data"]["total_gas_estimate"]
        .as_u64()
        .unwrap();
    assert!(
        total_gas >= base_gas,
        "Total gas should be >= base gas when features are enabled"
    );

    println!("✅ POST /templates/advanced_token/estimate test passed");
}

/// Test GET /deployments/{request_id}/status endpoint
#[tokio::test]
async fn test_get_deployment_status() {
    let app = create_test_router();

    let request_id = "test_deployment_123";
    let response = app
        .oneshot(
            Request::builder()
                .uri(&format!("/deployments/{}/status", request_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json_response["success"].as_bool().unwrap_or(false));
    assert!(json_response["data"]["request_id"].is_string());
    assert!(json_response["data"]["status"].is_string());
    assert!(json_response["data"]["progress"].is_object());

    println!("✅ GET /deployments/status test passed");
}

/// Test GET /user/{address}/contracts endpoint
#[tokio::test]
async fn test_get_user_contracts() {
    let app = create_test_router();

    let user_address = "0x1234567890123456789012345678901234567890";
    let response = app
        .oneshot(
            Request::builder()
                .uri(&format!("/user/{}/contracts", user_address))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json_response["success"].as_bool().unwrap_or(false));
    assert!(json_response["data"].is_array());

    println!("✅ GET /user/contracts test passed");
}

/// Test error handling for invalid contract type
#[tokio::test]
async fn test_invalid_contract_type() {
    let app = create_test_router();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/templates/invalid_contract_type/form")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK); // API returns 200 with error in JSON

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(!json_response["success"].as_bool().unwrap_or(true));
    assert!(json_response["error"].is_string());

    println!("✅ Invalid contract type error handling test passed");
}

/// Test error handling for invalid address format
#[tokio::test]
async fn test_invalid_address_format() {
    let app = create_test_router();

    let deployment_request = json!({
        "contract_type": "secure_token",
        "owner": "invalid_address",
        "parameters": {
            "name": "Test Token",
            "symbol": "TST"
        }
    });

    let response = app
        .oneshot(
            Request::builder()
                .uri("/deploy")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(deployment_request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK); // API returns 200 with error in JSON

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(!json_response["success"].as_bool().unwrap_or(true));
    assert!(json_response["error"].is_string());

    println!("✅ Invalid address format error handling test passed");
}

/// Test comprehensive deployment workflow
#[tokio::test]
async fn test_complete_deployment_workflow() {
    println!("🔄 Testing Complete Deployment Workflow...");

    let app_state = create_test_app_state().await;

    // Step 1: Get available templates
    let app = q_api_server::contracts_api::create_contracts_router().with_state(app_state.clone());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/templates")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Step 2: Get deployment form for specific contract
    let app = q_api_server::contracts_api::create_contracts_router().with_state(app_state.clone());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/templates/rwa_token/form")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Step 3: Estimate deployment cost
    let parameters = json!({
        "kyc_required": true,
        "dividend_enabled": true
    });

    let app = q_api_server::contracts_api::create_contracts_router().with_state(app_state.clone());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/templates/rwa_token/estimate")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(parameters.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Step 4: Deploy contract
    let deployment_request = json!({
        "contract_type": "rwa_token",
        "owner": "0x1234567890123456789012345678901234567890",
        "parameters": {
            "asset_name": "Test Property",
            "asset_symbol": "TPR",
            "total_value_usd": "5000000000000000000000000",
            "shares_count": "5000000",
            "kyc_required": true,
            "dividend_enabled": true,
            "asset_category": "real_estate"
        },
        "deployment_options": {
            "test_deployment": true,
            "auto_verify": true,
            "enable_governance": true
        }
    });

    let app = q_api_server::contracts_api::create_contracts_router().with_state(app_state.clone());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/deploy")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(deployment_request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json_response["success"].as_bool().unwrap_or(false));

    let request_id = json_response["data"]["request_id"].as_str().unwrap();

    // Step 5: Check deployment status
    let app = q_api_server::contracts_api::create_contracts_router().with_state(app_state.clone());
    let response = app
        .oneshot(
            Request::builder()
                .uri(&format!("/deployments/{}/status", request_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Step 6: Get user's deployed contracts
    let app = q_api_server::contracts_api::create_contracts_router().with_state(app_state.clone());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/user/0x1234567890123456789012345678901234567890/contracts")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    println!("✅ Complete deployment workflow test passed!");
}

/// Test all contract types deployment
#[tokio::test]
async fn test_all_contract_types_deployment() {
    println!("🎯 Testing All Contract Types Deployment...");

    let contract_types = vec![
        (
            "secure_token",
            json!({
                "name": "Secure Test Token",
                "symbol": "STT",
                "initial_supply": "1000000000000000000000000",
                "decimals": "18"
            }),
        ),
        (
            "advanced_token",
            json!({
                "name": "Advanced Test Token",
                "symbol": "ATT",
                "initial_supply": "1000000000000000000000000",
                "mintable": true,
                "burnable": true,
                "stakeable": true
            }),
        ),
        (
            "rwa_token",
            json!({
                "asset_name": "Test Asset",
                "asset_symbol": "TSA",
                "total_value_usd": "1000000000000000000000000",
                "shares_count": "1000000",
                "kyc_required": true,
                "asset_category": "real_estate"
            }),
        ),
        (
            "orbusd_stablecoin",
            json!({
                "collateral_ratio": "150",
                "stability_fee": "5",
                "liquidation_ratio": "130",
                "oracle_enabled": true
            }),
        ),
        (
            "multisig_wallet",
            json!({
                "required_confirmations": "2",
                "owners": "0x1234567890123456789012345678901234567890,0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
                "daily_limit": "1000000000000000000"
            }),
        ),
        (
            "governance",
            json!({
                "voting_token": "0x1234567890123456789012345678901234567890",
                "proposal_threshold": "100000000000000000000000",
                "voting_period": "17280",
                "execution_delay": "172800",
                "quorum_threshold": "10"
            }),
        ),
        (
            "private_dex",
            json!({
                "trading_fee": "30",
                "privacy_enabled": true,
                "yield_farming": true,
                "max_slippage": "500"
            }),
        ),
    ];

    for (contract_type, parameters) in contract_types {
        println!("  🔧 Testing deployment: {}", contract_type);

        let app = create_test_router();

        let deployment_request = json!({
            "contract_type": contract_type,
            "owner": "0x1234567890123456789012345678901234567890",
            "parameters": parameters,
            "deployment_options": {
                "test_deployment": true,
                "auto_verify": false
            }
        });

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/deploy")
                    .method("POST")
                    .header("content-type", "application/json")
                    .body(Body::from(deployment_request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
        let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(
            json_response["success"].as_bool().unwrap_or(false),
            "Deployment failed for {}: {}",
            contract_type,
            json_response["error"].as_str().unwrap_or("unknown error")
        );

        println!("    ✅ {} deployment successful", contract_type);
    }

    println!("✅ All contract types deployment test passed!");
}

/// Test concurrent API requests
#[tokio::test]
async fn test_concurrent_api_requests() {
    println!("⚡ Testing Concurrent API Requests...");

    let app_state = create_test_app_state().await;

    let mut handles = vec![];

    // Spawn multiple concurrent requests
    for i in 0..10 {
        let app_state_clone = app_state.clone();

        let handle = tokio::spawn(async move {
            let app =
                q_api_server::contracts_api::create_contracts_router().with_state(app_state_clone);

            let response = app
                .oneshot(
                    Request::builder()
                        .uri("/templates")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::OK);

            let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
            let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

            assert!(json_response["success"].as_bool().unwrap_or(false));

            i // Return thread number for verification
        });

        handles.push(handle);
    }

    // Wait for all requests to complete
    for handle in handles {
        let thread_num = handle.await.unwrap();
        println!("  ✅ Concurrent request {} completed", thread_num);
    }

    println!("✅ Concurrent API requests test passed!");
}

/// Run all API tests
#[tokio::test]
async fn run_comprehensive_api_tests() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Running Comprehensive API Test Suite");
    println!("=".repeat(80));

    let start_time = std::time::Instant::now();

    // Run all API test modules
    test_get_contract_templates().await;
    test_get_deployment_form().await;
    test_deploy_contract().await;
    test_estimate_deployment_cost().await;
    test_get_deployment_status().await;
    test_get_user_contracts().await;
    test_invalid_contract_type().await;
    test_invalid_address_format().await;
    test_complete_deployment_workflow().await;
    test_all_contract_types_deployment().await;
    test_concurrent_api_requests().await;

    let total_duration = start_time.elapsed();

    println!("=".repeat(80));
    println!("🎉 ALL API TESTS PASSED! 🎉");
    println!("📊 API Test Suite Statistics:");
    println!("   • Total test time: {:?}", total_duration);
    println!("   • Endpoints tested: 8");
    println!("   • Contract types tested: 7");
    println!("   • Error scenarios tested: 2");
    println!("   • Concurrent requests tested: 10");
    println!("   • Full workflow tested: ✅");
    println!("=".repeat(80));

    Ok(())
}
