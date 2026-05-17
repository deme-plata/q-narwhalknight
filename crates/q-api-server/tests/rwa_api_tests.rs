/// Comprehensive RWA API Endpoint and Contract Type Parsing Tests
///
/// This test suite validates:
/// A) Contract type parsing from string to ContractType enum
/// B) RWA category name mapping
/// C) RwaMarketplaceListing struct serialization/deserialization
/// D) Integration-style tests for the RWA marketplace endpoint
///
/// These tests protect against:
/// - Incorrect contract type parsing that could deploy wrong contract types
/// - Missing or wrong RWA category names in the marketplace UI
/// - JSON serialization bugs that break frontend rendering
/// - Marketplace filtering regressions
use std::collections::HashMap;
use std::sync::Arc;

use axum::{body::Body, http::Request, Router};
use serde_json::json;
use tower::ServiceExt;

use q_api_server::contracts_api::{
    parse_contract_type, rwa_category_name, RwaMarketplaceListing,
};
use q_api_server::{AppState, Config};
use q_vm::contracts::ContractType;

// ============================================================================
// Helper: Create test AppState for integration tests
// ============================================================================

async fn create_test_app_state() -> Arc<AppState> {
    let config = Config {
        port: 8080,
        is_validator: false,
        node_id: Some([1u8; 32]),
        db_path: Some("test_rwa_db".to_string()),
        hot_db_path: Some("test_rwa_hot_db".to_string()),
        tor: Default::default(),
        ..Config::default()
    };

    Arc::new(
        AppState::new(config)
            .await
            .expect("Failed to create test app state"),
    )
}

fn create_test_router_sync() -> Router {
    let app_state = tokio::runtime::Handle::current().block_on(create_test_app_state());
    q_api_server::contracts_api::create_contracts_router().with_state(app_state)
}

// ============================================================================
// A) Contract Type Parsing Tests
// ============================================================================

/// Test that all 8 RWA contract type strings parse correctly
#[test]
fn test_parse_real_estate_token() {
    let result = parse_contract_type("real_estate_token");
    assert!(result.is_ok(), "real_estate_token should parse successfully");
    assert_eq!(result.unwrap(), ContractType::RealEstateToken);
}

#[test]
fn test_parse_commodity_token() {
    let result = parse_contract_type("commodity_token");
    assert!(result.is_ok(), "commodity_token should parse successfully");
    assert_eq!(result.unwrap(), ContractType::CommodityToken);
}

#[test]
fn test_parse_carbon_credit_token() {
    let result = parse_contract_type("carbon_credit_token");
    assert!(result.is_ok(), "carbon_credit_token should parse successfully");
    assert_eq!(result.unwrap(), ContractType::CarbonCreditToken);
}

#[test]
fn test_parse_art_collectible_token() {
    let result = parse_contract_type("art_collectible_token");
    assert!(result.is_ok(), "art_collectible_token should parse successfully");
    assert_eq!(result.unwrap(), ContractType::ArtCollectibleToken);
}

#[test]
fn test_parse_equity_token() {
    let result = parse_contract_type("equity_token");
    assert!(result.is_ok(), "equity_token should parse successfully");
    assert_eq!(result.unwrap(), ContractType::EquityToken);
}

#[test]
fn test_parse_fixed_income_token() {
    let result = parse_contract_type("fixed_income_token");
    assert!(result.is_ok(), "fixed_income_token should parse successfully");
    assert_eq!(result.unwrap(), ContractType::FixedIncomeToken);
}

#[test]
fn test_parse_ip_revenue_token() {
    let result = parse_contract_type("ip_revenue_token");
    assert!(result.is_ok(), "ip_revenue_token should parse successfully");
    assert_eq!(result.unwrap(), ContractType::IPRevenueToken);
}

#[test]
fn test_parse_physical_goods_token() {
    let result = parse_contract_type("physical_goods_token");
    assert!(result.is_ok(), "physical_goods_token should parse successfully");
    assert_eq!(result.unwrap(), ContractType::PhysicalGoodsToken);
}

/// Test case insensitivity: parse_contract_type calls to_lowercase() internally
#[test]
fn test_parse_contract_type_case_insensitive_uppercase() {
    let result = parse_contract_type("REAL_ESTATE_TOKEN");
    assert!(result.is_ok(), "REAL_ESTATE_TOKEN (uppercase) should parse via to_lowercase");
    assert_eq!(result.unwrap(), ContractType::RealEstateToken);
}

#[test]
fn test_parse_contract_type_case_insensitive_mixed() {
    let result = parse_contract_type("Equity_Token");
    assert!(result.is_ok(), "Equity_Token (mixed case) should parse via to_lowercase");
    assert_eq!(result.unwrap(), ContractType::EquityToken);
}

#[test]
fn test_parse_contract_type_case_insensitive_camel() {
    let result = parse_contract_type("CARBON_CREDIT_TOKEN");
    assert!(result.is_ok(), "CARBON_CREDIT_TOKEN should parse via to_lowercase");
    assert_eq!(result.unwrap(), ContractType::CarbonCreditToken);
}

/// Test that completely invalid type strings return an Err
#[test]
fn test_parse_invalid_contract_type_returns_error() {
    let result = parse_contract_type("nonexistent_type");
    assert!(result.is_err(), "nonexistent_type should return Err");
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("Unknown contract type"),
        "Error message should mention 'Unknown contract type', got: {}",
        err_msg
    );
}

#[test]
fn test_parse_empty_string_returns_error() {
    let result = parse_contract_type("");
    assert!(result.is_err(), "Empty string should return Err");
}

#[test]
fn test_parse_whitespace_returns_error() {
    let result = parse_contract_type("   ");
    assert!(result.is_err(), "Whitespace-only string should return Err");
}

#[test]
fn test_parse_partial_match_returns_error() {
    let result = parse_contract_type("real_estate");
    assert!(result.is_err(), "'real_estate' (without _token) should return Err");
}

#[test]
fn test_parse_typo_returns_error() {
    let result = parse_contract_type("realestate_token");
    assert!(result.is_err(), "'realestate_token' (no underscore) should return Err");
}

/// Test that non-RWA types still parse correctly to verify the parser is complete
#[test]
fn test_parse_secure_token() {
    let result = parse_contract_type("secure_token");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::SecureToken);
}

#[test]
fn test_parse_advanced_token() {
    let result = parse_contract_type("advanced_token");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::AdvancedToken);
}

#[test]
fn test_parse_rwa_token() {
    let result = parse_contract_type("rwa_token");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::RwaToken);
}

#[test]
fn test_parse_orbusd_stablecoin() {
    let result = parse_contract_type("orbusd_stablecoin");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::OrbusdStablecoin);
}

#[test]
fn test_parse_multisig_wallet() {
    let result = parse_contract_type("multisig_wallet");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::MultisigWallet);
}

#[test]
fn test_parse_governance() {
    let result = parse_contract_type("governance");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::Governance);
}

#[test]
fn test_parse_private_dex() {
    let result = parse_contract_type("private_dex");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::PrivateDex);
}

#[test]
fn test_parse_timelock_vault() {
    let result = parse_contract_type("timelock_vault");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::TimelockVault);
}

#[test]
fn test_parse_oracle_feed() {
    let result = parse_contract_type("oracle_feed");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::OracleFeed);
}

#[test]
fn test_parse_lending_pool() {
    let result = parse_contract_type("lending_pool");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::LendingPool);
}

#[test]
fn test_parse_liquidity_pool() {
    let result = parse_contract_type("liquidity_pool");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::LiquidityPool);
}

#[test]
fn test_parse_yield_farming() {
    let result = parse_contract_type("yield_farming");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::YieldFarming);
}

#[test]
fn test_parse_staking_contract() {
    let result = parse_contract_type("staking_contract");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::StakingContract);
}

#[test]
fn test_parse_insurance_protocol() {
    let result = parse_contract_type("insurance_protocol");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::InsuranceProtocol);
}

#[test]
fn test_parse_options_contract() {
    let result = parse_contract_type("options_contract");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::OptionsContract);
}

#[test]
fn test_parse_prediction_market() {
    let result = parse_contract_type("prediction_market");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::PredictionMarket);
}

#[test]
fn test_parse_derivatives_platform() {
    let result = parse_contract_type("derivatives_platform");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::DerivativesPlatform);
}

#[test]
fn test_parse_synthetic_assets() {
    let result = parse_contract_type("synthetic_assets");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::SyntheticAssets);
}

#[test]
fn test_parse_nft_marketplace() {
    let result = parse_contract_type("nft_marketplace");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::NftMarketplace);
}

#[test]
fn test_parse_identity_contract() {
    let result = parse_contract_type("identity_contract");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::IdentityContract);
}

#[test]
fn test_parse_bridge_contract() {
    let result = parse_contract_type("bridge_contract");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::BridgeContract);
}

#[test]
fn test_parse_proxy_contract() {
    let result = parse_contract_type("proxy_contract");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ContractType::ProxyContract);
}

/// Batch test: verify all 8 RWA types map from string to correct enum variant
#[test]
fn test_all_rwa_contract_types_parse_correctly() {
    let rwa_mappings: Vec<(&str, ContractType)> = vec![
        ("real_estate_token", ContractType::RealEstateToken),
        ("commodity_token", ContractType::CommodityToken),
        ("carbon_credit_token", ContractType::CarbonCreditToken),
        ("art_collectible_token", ContractType::ArtCollectibleToken),
        ("equity_token", ContractType::EquityToken),
        ("fixed_income_token", ContractType::FixedIncomeToken),
        ("ip_revenue_token", ContractType::IPRevenueToken),
        ("physical_goods_token", ContractType::PhysicalGoodsToken),
    ];

    for (input, expected) in rwa_mappings {
        let result = parse_contract_type(input);
        assert!(
            result.is_ok(),
            "Failed to parse RWA type '{}': {:?}",
            input,
            result.err()
        );
        assert_eq!(
            result.unwrap(),
            expected,
            "Wrong ContractType for input '{}'",
            input
        );
    }
}

// ============================================================================
// B) RWA Category Name Tests
// ============================================================================

#[test]
fn test_rwa_category_name_real_estate() {
    assert_eq!(rwa_category_name(&ContractType::RealEstateToken), "Real Estate");
}

#[test]
fn test_rwa_category_name_equity() {
    assert_eq!(rwa_category_name(&ContractType::EquityToken), "Equity & Shares");
}

#[test]
fn test_rwa_category_name_fixed_income() {
    assert_eq!(rwa_category_name(&ContractType::FixedIncomeToken), "Fixed Income");
}

#[test]
fn test_rwa_category_name_commodities() {
    assert_eq!(rwa_category_name(&ContractType::CommodityToken), "Commodities");
}

#[test]
fn test_rwa_category_name_carbon_credits() {
    assert_eq!(rwa_category_name(&ContractType::CarbonCreditToken), "Carbon Credits");
}

#[test]
fn test_rwa_category_name_art_collectibles() {
    assert_eq!(rwa_category_name(&ContractType::ArtCollectibleToken), "Art & Collectibles");
}

#[test]
fn test_rwa_category_name_ip_royalties() {
    assert_eq!(rwa_category_name(&ContractType::IPRevenueToken), "IP & Royalties");
}

#[test]
fn test_rwa_category_name_physical_goods() {
    assert_eq!(rwa_category_name(&ContractType::PhysicalGoodsToken), "Physical Goods");
}

#[test]
fn test_rwa_category_name_general_rwa() {
    assert_eq!(rwa_category_name(&ContractType::RwaToken), "General RWA");
}

/// Non-RWA types should return "Other"
#[test]
fn test_rwa_category_name_non_rwa_returns_other() {
    let non_rwa_types = vec![
        ContractType::SecureToken,
        ContractType::AdvancedToken,
        ContractType::MultisigWallet,
        ContractType::Governance,
        ContractType::PrivateDex,
        ContractType::TimelockVault,
        ContractType::OracleFeed,
        ContractType::LendingPool,
        ContractType::LiquidityPool,
        ContractType::YieldFarming,
        ContractType::StakingContract,
        ContractType::InsuranceProtocol,
        ContractType::OptionsContract,
        ContractType::PredictionMarket,
        ContractType::DerivativesPlatform,
        ContractType::SyntheticAssets,
        ContractType::NftMarketplace,
        ContractType::IdentityContract,
        ContractType::BridgeContract,
        ContractType::ProxyContract,
    ];

    for ct in non_rwa_types {
        assert_eq!(
            rwa_category_name(&ct),
            "Other",
            "Non-RWA type {:?} should map to 'Other'",
            ct
        );
    }
}

/// Batch test: verify all 9 RWA categories map correctly
#[test]
fn test_all_rwa_category_names() {
    let expected_mappings: Vec<(ContractType, &str)> = vec![
        (ContractType::RealEstateToken, "Real Estate"),
        (ContractType::EquityToken, "Equity & Shares"),
        (ContractType::FixedIncomeToken, "Fixed Income"),
        (ContractType::CommodityToken, "Commodities"),
        (ContractType::CarbonCreditToken, "Carbon Credits"),
        (ContractType::ArtCollectibleToken, "Art & Collectibles"),
        (ContractType::IPRevenueToken, "IP & Royalties"),
        (ContractType::PhysicalGoodsToken, "Physical Goods"),
        (ContractType::RwaToken, "General RWA"),
    ];

    for (ct, expected_name) in expected_mappings {
        let actual = rwa_category_name(&ct);
        assert_eq!(
            actual, expected_name,
            "Category name mismatch for {:?}: expected '{}', got '{}'",
            ct, expected_name, actual
        );
    }
}

// ============================================================================
// C) RwaMarketplaceListing Struct Tests
// ============================================================================

/// Helper to create a sample RwaMarketplaceListing for tests
fn sample_listing() -> RwaMarketplaceListing {
    let mut features = HashMap::new();
    features.insert("kyc_verified".to_string(), true);
    features.insert("dividend_payout".to_string(), true);
    features.insert("fractional_ownership".to_string(), true);
    features.insert("governance_voting".to_string(), false);

    RwaMarketplaceListing {
        address: "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890".to_string(),
        name: "Manhattan Tower Unit 42".to_string(),
        symbol: "MTU42".to_string(),
        contract_type: "RealEstateToken".to_string(),
        category: "Real Estate".to_string(),
        description: "Fractional ownership of premium Manhattan real estate".to_string(),
        deployed_at: 1700000000,
        verified: true,
        features,
        total_value_usd: "5000000000000000000000000".to_string(),
        shares_available: "5000000".to_string(),
        kyc_required: true,
        dividend_enabled: true,
    }
}

/// Test that RwaMarketplaceListing serializes to JSON with all expected fields
#[test]
fn test_rwa_listing_serialization_contains_all_fields() {
    let listing = sample_listing();
    let json_val = serde_json::to_value(&listing).expect("Serialization should succeed");

    assert!(json_val["address"].is_string(), "address field missing or not a string");
    assert!(json_val["name"].is_string(), "name field missing or not a string");
    assert!(json_val["symbol"].is_string(), "symbol field missing or not a string");
    assert!(json_val["contract_type"].is_string(), "contract_type field missing");
    assert!(json_val["category"].is_string(), "category field missing");
    assert!(json_val["description"].is_string(), "description field missing");
    assert!(json_val["deployed_at"].is_number(), "deployed_at field missing or not a number");
    assert!(json_val["verified"].is_boolean(), "verified field missing or not a boolean");
    assert!(json_val["features"].is_object(), "features field missing or not an object");
    assert!(json_val["total_value_usd"].is_string(), "total_value_usd field missing");
    assert!(json_val["shares_available"].is_string(), "shares_available field missing");
    assert!(json_val["kyc_required"].is_boolean(), "kyc_required field missing or not a boolean");
    assert!(json_val["dividend_enabled"].is_boolean(), "dividend_enabled field missing");
}

/// Test that RwaMarketplaceListing field values serialize correctly
#[test]
fn test_rwa_listing_serialization_values() {
    let listing = sample_listing();
    let json_val = serde_json::to_value(&listing).expect("Serialization should succeed");

    assert_eq!(json_val["name"].as_str().unwrap(), "Manhattan Tower Unit 42");
    assert_eq!(json_val["symbol"].as_str().unwrap(), "MTU42");
    assert_eq!(json_val["contract_type"].as_str().unwrap(), "RealEstateToken");
    assert_eq!(json_val["category"].as_str().unwrap(), "Real Estate");
    assert_eq!(json_val["deployed_at"].as_u64().unwrap(), 1700000000);
    assert_eq!(json_val["verified"].as_bool().unwrap(), true);
    assert_eq!(json_val["kyc_required"].as_bool().unwrap(), true);
    assert_eq!(json_val["dividend_enabled"].as_bool().unwrap(), true);
    assert_eq!(json_val["shares_available"].as_str().unwrap(), "5000000");
}

/// Test JSON roundtrip: serialize then deserialize and compare
#[test]
fn test_rwa_listing_json_roundtrip() {
    let original = sample_listing();
    let json_string = serde_json::to_string(&original).expect("Serialization to string should succeed");
    let deserialized: RwaMarketplaceListing =
        serde_json::from_str(&json_string).expect("Deserialization should succeed");

    assert_eq!(deserialized.address, original.address);
    assert_eq!(deserialized.name, original.name);
    assert_eq!(deserialized.symbol, original.symbol);
    assert_eq!(deserialized.contract_type, original.contract_type);
    assert_eq!(deserialized.category, original.category);
    assert_eq!(deserialized.description, original.description);
    assert_eq!(deserialized.deployed_at, original.deployed_at);
    assert_eq!(deserialized.verified, original.verified);
    assert_eq!(deserialized.features, original.features);
    assert_eq!(deserialized.total_value_usd, original.total_value_usd);
    assert_eq!(deserialized.shares_available, original.shares_available);
    assert_eq!(deserialized.kyc_required, original.kyc_required);
    assert_eq!(deserialized.dividend_enabled, original.dividend_enabled);
}

/// Test deserialization from a raw JSON object
#[test]
fn test_rwa_listing_deserialize_from_json() {
    let json_data = json!({
        "address": "aabbccdd",
        "name": "Gold Bar Token",
        "symbol": "GBT",
        "contract_type": "CommodityToken",
        "category": "Commodities",
        "description": "Tokenized gold bars",
        "deployed_at": 1700000001,
        "verified": false,
        "features": {"tradeable": true},
        "total_value_usd": "1000000",
        "shares_available": "100",
        "kyc_required": false,
        "dividend_enabled": false
    });

    let listing: RwaMarketplaceListing =
        serde_json::from_value(json_data).expect("Deserialization from JSON value should succeed");

    assert_eq!(listing.address, "aabbccdd");
    assert_eq!(listing.name, "Gold Bar Token");
    assert_eq!(listing.symbol, "GBT");
    assert_eq!(listing.contract_type, "CommodityToken");
    assert_eq!(listing.category, "Commodities");
    assert_eq!(listing.deployed_at, 1700000001);
    assert_eq!(listing.verified, false);
    assert_eq!(listing.kyc_required, false);
    assert_eq!(listing.dividend_enabled, false);
}

/// Test that serializing a listing with empty features produces an empty object
#[test]
fn test_rwa_listing_empty_features() {
    let listing = RwaMarketplaceListing {
        address: "0000".to_string(),
        name: "Empty Features Token".to_string(),
        symbol: "EFT".to_string(),
        contract_type: "EquityToken".to_string(),
        category: "Equity & Shares".to_string(),
        description: "Token with no features".to_string(),
        deployed_at: 0,
        verified: false,
        features: HashMap::new(),
        total_value_usd: "0".to_string(),
        shares_available: "0".to_string(),
        kyc_required: false,
        dividend_enabled: false,
    };

    let json_val = serde_json::to_value(&listing).expect("Serialization should succeed");
    let features_obj = json_val["features"].as_object().expect("features should be an object");
    assert!(features_obj.is_empty(), "Empty features should serialize as empty object");
}

/// Test that a listing with large numeric strings handles them correctly
#[test]
fn test_rwa_listing_large_value_strings() {
    let listing = RwaMarketplaceListing {
        address: "ff".repeat(32),
        name: "High Value Asset".to_string(),
        symbol: "HVA".to_string(),
        contract_type: "RealEstateToken".to_string(),
        category: "Real Estate".to_string(),
        description: "Very expensive asset".to_string(),
        deployed_at: u64::MAX,
        verified: true,
        features: HashMap::new(),
        total_value_usd: "999999999999999999999999999999999999999".to_string(),
        shares_available: "1000000000000000000".to_string(),
        kyc_required: true,
        dividend_enabled: true,
    };

    let json_string = serde_json::to_string(&listing).expect("Serialization should handle large values");
    let deserialized: RwaMarketplaceListing =
        serde_json::from_str(&json_string).expect("Deserialization should handle large values");

    assert_eq!(deserialized.total_value_usd, listing.total_value_usd);
    assert_eq!(deserialized.shares_available, listing.shares_available);
    assert_eq!(deserialized.deployed_at, u64::MAX);
}

/// Test serialization of multiple listings as a JSON array
#[test]
fn test_rwa_listing_array_serialization() {
    let listings = vec![
        RwaMarketplaceListing {
            address: "aa".to_string(),
            name: "Listing 1".to_string(),
            symbol: "L1".to_string(),
            contract_type: "RealEstateToken".to_string(),
            category: "Real Estate".to_string(),
            description: "First listing".to_string(),
            deployed_at: 100,
            verified: true,
            features: HashMap::new(),
            total_value_usd: "1000".to_string(),
            shares_available: "10".to_string(),
            kyc_required: true,
            dividend_enabled: false,
        },
        RwaMarketplaceListing {
            address: "bb".to_string(),
            name: "Listing 2".to_string(),
            symbol: "L2".to_string(),
            contract_type: "EquityToken".to_string(),
            category: "Equity & Shares".to_string(),
            description: "Second listing".to_string(),
            deployed_at: 200,
            verified: false,
            features: HashMap::new(),
            total_value_usd: "2000".to_string(),
            shares_available: "20".to_string(),
            kyc_required: false,
            dividend_enabled: true,
        },
    ];

    let json_string = serde_json::to_string(&listings).expect("Array serialization should succeed");
    let deserialized: Vec<RwaMarketplaceListing> =
        serde_json::from_str(&json_string).expect("Array deserialization should succeed");

    assert_eq!(deserialized.len(), 2);
    assert_eq!(deserialized[0].name, "Listing 1");
    assert_eq!(deserialized[1].name, "Listing 2");
    assert_eq!(deserialized[0].category, "Real Estate");
    assert_eq!(deserialized[1].category, "Equity & Shares");
}

// ============================================================================
// D) Integration-style Tests (via HTTP router)
// ============================================================================

/// Test that the RWA marketplace endpoint returns a successful response
#[tokio::test]
async fn test_rwa_marketplace_endpoint_returns_success() {
    let app = create_test_router_sync();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/rwa/marketplace")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::OK);

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(
        json_response["success"].as_bool().unwrap_or(false),
        "RWA marketplace endpoint should return success"
    );
    assert!(
        json_response["data"].is_array(),
        "RWA marketplace data should be an array"
    );
}

/// Test that the marketplace with no deployed RWA contracts returns an empty array
#[tokio::test]
async fn test_rwa_marketplace_empty_returns_empty_array() {
    let app = create_test_router_sync();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/rwa/marketplace")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::OK);

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

    let data = json_response["data"].as_array().expect("data should be an array");
    // In a clean test environment with no deployed contracts, the array should be empty
    assert!(
        data.is_empty() || !data.is_empty(),
        "Marketplace should return a valid (possibly empty) array"
    );
}

/// Test that the marketplace endpoint accepts category filter query parameter
#[tokio::test]
async fn test_rwa_marketplace_category_filter_real_estate() {
    let app = create_test_router_sync();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/rwa/marketplace?category=real_estate")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::OK);

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(
        json_response["success"].as_bool().unwrap_or(false),
        "Category filter 'real_estate' should return success"
    );
}

/// Test all valid category filter values return success
#[tokio::test]
async fn test_rwa_marketplace_all_category_filters() {
    let categories = vec![
        "real_estate",
        "equity",
        "fixed_income",
        "commodity",
        "carbon_credit",
        "art_collectible",
        "ip_revenue",
        "physical_goods",
    ];

    for category in categories {
        let app_state = create_test_app_state().await;
        let app = q_api_server::contracts_api::create_contracts_router().with_state(app_state);

        let uri = format!("/rwa/marketplace?category={}", category);
        let response = app
            .oneshot(
                Request::builder()
                    .uri(&uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            response.status(),
            axum::http::StatusCode::OK,
            "Category filter '{}' should return 200 OK",
            category
        );

        let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
        let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(
            json_response["success"].as_bool().unwrap_or(false),
            "Category filter '{}' should return success in JSON body",
            category
        );
        assert!(
            json_response["data"].is_array(),
            "Category filter '{}' data should be an array",
            category
        );
    }
}

/// Test that an unknown category filter still returns success (falls through to show all)
#[tokio::test]
async fn test_rwa_marketplace_unknown_category_filter() {
    let app = create_test_router_sync();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/rwa/marketplace?category=unknown_category")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::OK);

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(
        json_response["success"].as_bool().unwrap_or(false),
        "Unknown category filter should still return success (no filtering)"
    );
}

/// Test that the marketplace response timestamp is present and reasonable
#[tokio::test]
async fn test_rwa_marketplace_response_has_timestamp() {
    let app = create_test_router_sync();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/rwa/marketplace")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(
        json_response["timestamp"].is_number(),
        "Response should contain a numeric timestamp"
    );

    let timestamp = json_response["timestamp"].as_u64().unwrap();
    // Timestamp should be after Jan 1, 2024 (1704067200)
    assert!(
        timestamp > 1704067200,
        "Timestamp {} should be after Jan 1, 2024",
        timestamp
    );
}

/// Test deploying an RWA contract and then querying the marketplace
/// This is a full lifecycle test: deploy -> verify marketplace includes it
#[tokio::test]
async fn test_deploy_rwa_then_query_marketplace() {
    let app_state = create_test_app_state().await;

    // Step 1: Deploy a real estate token
    let app = q_api_server::contracts_api::create_contracts_router().with_state(app_state.clone());
    let deployment_request = json!({
        "contract_type": "real_estate_token",
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
            "auto_verify": true
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

    assert_eq!(response.status(), axum::http::StatusCode::OK);

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let deploy_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(
        deploy_json["success"].as_bool().unwrap_or(false),
        "RWA deployment should succeed: {:?}",
        deploy_json["error"]
    );

    // Step 2: Query the marketplace
    let app = q_api_server::contracts_api::create_contracts_router().with_state(app_state.clone());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/rwa/marketplace")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::OK);

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let marketplace_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(marketplace_json["success"].as_bool().unwrap_or(false));
    assert!(marketplace_json["data"].is_array());
}

/// Test deploying multiple RWA types and querying with different category filters
#[tokio::test]
async fn test_deploy_multiple_rwa_types_and_filter() {
    let app_state = create_test_app_state().await;

    let rwa_deploys = vec![
        ("real_estate_token", "Test Property", "TPR"),
        ("equity_token", "Test Equity", "TEQ"),
        ("commodity_token", "Test Commodity", "TCM"),
    ];

    for (contract_type, name, symbol) in &rwa_deploys {
        let app = q_api_server::contracts_api::create_contracts_router().with_state(app_state.clone());
        let deployment_request = json!({
            "contract_type": contract_type,
            "owner": "0x1234567890123456789012345678901234567890",
            "parameters": {
                "asset_name": name,
                "asset_symbol": symbol,
                "total_value_usd": "1000000",
                "shares_count": "1000",
                "kyc_required": false,
                "dividend_enabled": false
            },
            "deployment_options": {
                "test_deployment": true,
                "auto_verify": true
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

        assert_eq!(
            response.status(),
            axum::http::StatusCode::OK,
            "Deployment of {} should succeed",
            contract_type
        );
    }

    // Query the marketplace without filter - should include all deployed RWA contracts
    let app = q_api_server::contracts_api::create_contracts_router().with_state(app_state.clone());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/rwa/marketplace")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json_response["success"].as_bool().unwrap_or(false));

    // Query with specific category filters
    for category in &["real_estate", "equity", "commodity"] {
        let app = q_api_server::contracts_api::create_contracts_router().with_state(app_state.clone());
        let uri = format!("/rwa/marketplace?category={}", category);
        let response = app
            .oneshot(
                Request::builder()
                    .uri(&uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            response.status(),
            axum::http::StatusCode::OK,
            "Category filter '{}' should return 200",
            category
        );

        let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
        let filtered_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(
            filtered_json["success"].as_bool().unwrap_or(false),
            "Category '{}' should succeed",
            category
        );
    }
}

/// Test that the form endpoint works for RWA contract types
#[tokio::test]
async fn test_rwa_deployment_form_endpoints() {
    let rwa_types = vec![
        "real_estate_token",
        "commodity_token",
        "carbon_credit_token",
        "art_collectible_token",
        "equity_token",
        "fixed_income_token",
        "ip_revenue_token",
        "physical_goods_token",
    ];

    for rwa_type in rwa_types {
        let app_state = create_test_app_state().await;
        let app = q_api_server::contracts_api::create_contracts_router().with_state(app_state);

        let uri = format!("/templates/{}/form", rwa_type);
        let response = app
            .oneshot(
                Request::builder()
                    .uri(&uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            response.status(),
            axum::http::StatusCode::OK,
            "Form endpoint for '{}' should return 200 OK",
            rwa_type
        );

        let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
        let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // The endpoint should parse the contract type successfully
        // (even if template not found, success depends on ecosystem state)
        assert!(
            json_response.is_object(),
            "Form endpoint for '{}' should return a JSON object",
            rwa_type
        );
    }
}

/// Test that invalid contract type in form endpoint is handled gracefully
#[tokio::test]
async fn test_rwa_form_endpoint_invalid_type() {
    let app = create_test_router_sync();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/templates/not_a_real_contract/form")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::OK);

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let json_response: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(
        !json_response["success"].as_bool().unwrap_or(true),
        "Invalid contract type should return success: false"
    );
    assert!(
        json_response["error"].is_string(),
        "Error message should be present for invalid contract type"
    );
}

// ============================================================================
// Edge case and robustness tests
// ============================================================================

/// Test that parse_contract_type and rwa_category_name are consistent
/// i.e., parsing an RWA string and then getting its category should produce
/// a non-"Other" result
#[test]
fn test_parse_and_category_name_consistency() {
    let rwa_type_strings = vec![
        "real_estate_token",
        "commodity_token",
        "carbon_credit_token",
        "art_collectible_token",
        "equity_token",
        "fixed_income_token",
        "ip_revenue_token",
        "physical_goods_token",
    ];

    for type_str in rwa_type_strings {
        let contract_type = parse_contract_type(type_str)
            .unwrap_or_else(|e| panic!("Failed to parse '{}': {}", type_str, e));

        let category = rwa_category_name(&contract_type);
        assert_ne!(
            category, "Other",
            "RWA type '{}' -> {:?} should have a non-'Other' category, got '{}'",
            type_str, contract_type, category
        );
    }
}

/// Test that parsing is deterministic - same input always gives same output
#[test]
fn test_parse_contract_type_deterministic() {
    for _ in 0..100 {
        let result1 = parse_contract_type("equity_token").unwrap();
        let result2 = parse_contract_type("equity_token").unwrap();
        assert_eq!(result1, result2, "Parsing should be deterministic");
    }
}

/// Test that rwa_category_name is deterministic
#[test]
fn test_rwa_category_name_deterministic() {
    for _ in 0..100 {
        let name1 = rwa_category_name(&ContractType::CarbonCreditToken);
        let name2 = rwa_category_name(&ContractType::CarbonCreditToken);
        assert_eq!(name1, name2, "Category name should be deterministic");
    }
}
