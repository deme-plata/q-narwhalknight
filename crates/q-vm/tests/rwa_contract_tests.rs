/// Comprehensive Test Suite for 8 RWA (Real World Asset) Smart Contract Templates
///
/// Tests cover template loading, content validation, deployment parameters,
/// ABI functions, gas estimation, and ContractType enum correctness for:
/// 1. RealEstateToken
/// 2. CommodityToken
/// 3. CarbonCreditToken
/// 4. EquityToken
/// 5. FixedIncomeToken
/// 6. IPRevenueToken
/// 7. PhysicalGoodsToken
/// 8. ArtCollectibleToken
use anyhow::Result;

use q_vm::contracts::{
    ContractType, FormDefinition, OrobitSmartContractEcosystem, SmartContractTemplate,
};

// ============================================================================
// Helper functions
// ============================================================================

/// Create a fresh ecosystem instance (loads all templates internally)
async fn create_ecosystem() -> OrobitSmartContractEcosystem {
    OrobitSmartContractEcosystem::new()
        .await
        .expect("Failed to initialize OrobitSmartContractEcosystem")
}

/// Assert that a template has a deployment parameter with the given name
fn assert_has_param(template: &SmartContractTemplate, name: &str) {
    assert!(
        template
            .deployment_parameters
            .iter()
            .any(|p| p.name == name),
        "Template '{}' is missing deployment parameter '{}'",
        template.name,
        name
    );
}

/// Assert that a template's ABI contains a function with the given name
fn assert_has_abi_function(template: &SmartContractTemplate, func_name: &str) {
    assert!(
        template.abi.functions.iter().any(|f| f.name == func_name),
        "Template '{}' ABI is missing function '{}'",
        template.name,
        func_name
    );
}

/// All 8 RWA contract types
fn all_rwa_types() -> Vec<ContractType> {
    vec![
        ContractType::RealEstateToken,
        ContractType::CommodityToken,
        ContractType::CarbonCreditToken,
        ContractType::EquityToken,
        ContractType::FixedIncomeToken,
        ContractType::IPRevenueToken,
        ContractType::PhysicalGoodsToken,
        ContractType::ArtCollectibleToken,
    ]
}

// ============================================================================
// A) Template Loading Tests
// ============================================================================

#[tokio::test]
async fn test_load_all_contract_templates_includes_all_8_rwa_types() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let available = ecosystem.get_available_contracts().await;

    for rwa_type in all_rwa_types() {
        assert!(
            available.contains(&rwa_type),
            "load_all_contract_templates() did not register {:?}",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_load_real_estate_template() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::RealEstateToken)
        .await?;
    assert_eq!(template.contract_type, ContractType::RealEstateToken);
    assert_eq!(template.name, "Real Estate Token");
    Ok(())
}

#[tokio::test]
async fn test_load_commodity_template() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::CommodityToken)
        .await?;
    assert_eq!(template.contract_type, ContractType::CommodityToken);
    assert_eq!(template.name, "Commodity Token");
    Ok(())
}

#[tokio::test]
async fn test_load_carbon_credit_template() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::CarbonCreditToken)
        .await?;
    assert_eq!(template.contract_type, ContractType::CarbonCreditToken);
    assert_eq!(template.name, "Carbon Credit Token");
    Ok(())
}

#[tokio::test]
async fn test_load_equity_template() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::EquityToken)
        .await?;
    assert_eq!(template.contract_type, ContractType::EquityToken);
    assert_eq!(template.name, "Equity Token");
    Ok(())
}

#[tokio::test]
async fn test_load_fixed_income_template() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::FixedIncomeToken)
        .await?;
    assert_eq!(template.contract_type, ContractType::FixedIncomeToken);
    assert_eq!(template.name, "Fixed Income Token");
    Ok(())
}

#[tokio::test]
async fn test_load_ip_revenue_template() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::IPRevenueToken)
        .await?;
    assert_eq!(template.contract_type, ContractType::IPRevenueToken);
    assert_eq!(template.name, "IP Revenue Token");
    Ok(())
}

#[tokio::test]
async fn test_load_physical_goods_template() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::PhysicalGoodsToken)
        .await?;
    assert_eq!(template.contract_type, ContractType::PhysicalGoodsToken);
    assert_eq!(template.name, "Physical Goods Token");
    Ok(())
}

#[tokio::test]
async fn test_load_art_collectible_template() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::ArtCollectibleToken)
        .await?;
    assert_eq!(template.contract_type, ContractType::ArtCollectibleToken);
    assert_eq!(template.name, "Art & Collectible Token");
    Ok(())
}

#[tokio::test]
async fn test_templates_registered_in_template_registry() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let templates = ecosystem.contract_templates.read().await;

    for rwa_type in all_rwa_types() {
        assert!(
            templates.contains_key(&rwa_type),
            "{:?} not found in contract_templates registry",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_form_definitions_registered_for_all_rwa_types() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let forms = ecosystem.form_definitions.read().await;

    for rwa_type in all_rwa_types() {
        assert!(
            forms.contains_key(&rwa_type),
            "{:?} not found in form_definitions registry",
            rwa_type
        );
    }

    Ok(())
}

// ============================================================================
// B) Template Content Validation Tests
// ============================================================================

#[tokio::test]
async fn test_all_rwa_templates_have_correct_contract_type() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        assert_eq!(
            template.contract_type, rwa_type,
            "Template contract_type mismatch for {:?}",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_all_rwa_templates_have_non_empty_deployment_parameters() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        assert!(
            !template.deployment_parameters.is_empty(),
            "{:?} has empty deployment_parameters",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_all_rwa_templates_have_valid_form_config_with_sections() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        assert!(
            !template.form_config.sections.is_empty(),
            "{:?} form_config has no sections",
            rwa_type
        );
        assert!(
            !template.form_config.title.is_empty(),
            "{:?} form_config has empty title",
            rwa_type
        );
        assert!(
            !template.form_config.description.is_empty(),
            "{:?} form_config has empty description",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_all_rwa_templates_have_valid_abi_with_functions() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        assert!(
            !template.abi.functions.is_empty(),
            "{:?} ABI has no functions",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_all_rwa_templates_have_security_features_enabled() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        let sec = &template.security_features;
        assert!(
            sec.reentrancy_protection,
            "{:?} missing reentrancy_protection",
            rwa_type
        );
        assert!(
            sec.overflow_protection,
            "{:?} missing overflow_protection",
            rwa_type
        );
        assert!(
            sec.access_control,
            "{:?} missing access_control",
            rwa_type
        );
        assert!(sec.pausable, "{:?} missing pausable", rwa_type);
        assert!(sec.upgradeable, "{:?} missing upgradeable", rwa_type);
        assert!(
            sec.multisig_required,
            "{:?} missing multisig_required",
            rwa_type
        );
        assert!(
            sec.timelock_enabled,
            "{:?} missing timelock_enabled",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_all_rwa_templates_have_non_empty_description() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        assert!(
            !template.description.is_empty(),
            "{:?} has empty description",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_all_rwa_templates_have_version() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        assert!(
            !template.version.is_empty(),
            "{:?} has empty version",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_all_rwa_templates_have_non_empty_documentation() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        assert!(
            !template.documentation.overview.is_empty(),
            "{:?} has empty documentation overview",
            rwa_type
        );
        assert!(
            !template.documentation.usage_guide.is_empty(),
            "{:?} has empty documentation usage_guide",
            rwa_type
        );
        assert!(
            !template.documentation.security_considerations.is_empty(),
            "{:?} has empty documentation security_considerations",
            rwa_type
        );
        assert!(
            !template.documentation.api_reference.is_empty(),
            "{:?} has empty documentation api_reference",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_all_rwa_templates_have_at_least_one_example() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        assert!(
            !template.examples.is_empty(),
            "{:?} has no examples",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_all_rwa_templates_have_deployment_flow_steps() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        assert!(
            !template.form_config.deployment_flow.is_empty(),
            "{:?} has no deployment_flow steps",
            rwa_type
        );
    }

    Ok(())
}

// ============================================================================
// C) Deployment Parameter Tests
// ============================================================================

#[tokio::test]
async fn test_real_estate_deployment_parameters() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::RealEstateToken)
        .await?;

    assert_has_param(&template, "property_type");
    assert_has_param(&template, "location");
    assert_has_param(&template, "total_valuation_usd");
    assert_has_param(&template, "rental_yield_percent");
    assert_has_param(&template, "kyc_required");
    assert_has_param(&template, "property_name");
    assert_has_param(&template, "property_symbol");
    assert_has_param(&template, "total_shares");
    assert_has_param(&template, "occupancy_rate");
    assert_has_param(&template, "accredited_only");
    assert_has_param(&template, "dividend_enabled");
    assert_has_param(&template, "transfer_restrictions");

    // Verify property_type and location are required
    let property_type_param = template
        .deployment_parameters
        .iter()
        .find(|p| p.name == "property_type")
        .unwrap();
    assert!(property_type_param.required, "property_type should be required");

    let location_param = template
        .deployment_parameters
        .iter()
        .find(|p| p.name == "location")
        .unwrap();
    assert!(location_param.required, "location should be required");

    Ok(())
}

#[tokio::test]
async fn test_equity_deployment_parameters() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::EquityToken)
        .await?;

    assert_has_param(&template, "share_class");
    assert_has_param(&template, "dividend_schedule");
    assert_has_param(&template, "vesting_period_months");
    assert_has_param(&template, "voting_rights");
    assert_has_param(&template, "company_name");
    assert_has_param(&template, "ticker_symbol");
    assert_has_param(&template, "total_shares");
    assert_has_param(&template, "price_per_share_usd");
    assert_has_param(&template, "vesting_enabled");
    assert_has_param(&template, "lockup_period_days");
    assert_has_param(&template, "kyc_required");
    assert_has_param(&template, "accredited_only");

    // share_class should be required
    let share_class_param = template
        .deployment_parameters
        .iter()
        .find(|p| p.name == "share_class")
        .unwrap();
    assert!(share_class_param.required, "share_class should be required");

    Ok(())
}

#[tokio::test]
async fn test_fixed_income_deployment_parameters() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::FixedIncomeToken)
        .await?;

    assert_has_param(&template, "instrument_type");
    assert_has_param(&template, "face_value_usd");
    assert_has_param(&template, "coupon_rate_percent");
    assert_has_param(&template, "maturity_date");
    assert_has_param(&template, "credit_rating");
    assert_has_param(&template, "instrument_name");
    assert_has_param(&template, "instrument_symbol");
    assert_has_param(&template, "payment_frequency");
    assert_has_param(&template, "total_units");
    assert_has_param(&template, "callable");
    assert_has_param(&template, "convertible");
    assert_has_param(&template, "kyc_required");

    // instrument_type should be required
    let instrument_type_param = template
        .deployment_parameters
        .iter()
        .find(|p| p.name == "instrument_type")
        .unwrap();
    assert!(
        instrument_type_param.required,
        "instrument_type should be required"
    );

    // credit_rating should be required
    let credit_rating_param = template
        .deployment_parameters
        .iter()
        .find(|p| p.name == "credit_rating")
        .unwrap();
    assert!(
        credit_rating_param.required,
        "credit_rating should be required"
    );

    Ok(())
}

#[tokio::test]
async fn test_commodity_deployment_parameters() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::CommodityToken)
        .await?;

    assert_has_param(&template, "commodity_type");
    assert_has_param(&template, "unit_of_measurement");
    assert_has_param(&template, "storage_provider");
    assert_has_param(&template, "delivery_option");
    assert_has_param(&template, "commodity_name");
    assert_has_param(&template, "commodity_symbol");
    assert_has_param(&template, "quantity_per_token");
    assert_has_param(&template, "total_tokens");
    assert_has_param(&template, "storage_location");
    assert_has_param(&template, "insurance_enabled");
    assert_has_param(&template, "kyc_required");

    // commodity_type should be required
    let commodity_type_param = template
        .deployment_parameters
        .iter()
        .find(|p| p.name == "commodity_type")
        .unwrap();
    assert!(
        commodity_type_param.required,
        "commodity_type should be required"
    );

    // storage_provider should be required
    let storage_param = template
        .deployment_parameters
        .iter()
        .find(|p| p.name == "storage_provider")
        .unwrap();
    assert!(storage_param.required, "storage_provider should be required");

    Ok(())
}

#[tokio::test]
async fn test_carbon_credit_deployment_parameters() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::CarbonCreditToken)
        .await?;

    assert_has_param(&template, "credit_standard");
    assert_has_param(&template, "project_type");
    assert_has_param(&template, "vintage_year");
    assert_has_param(&template, "retirement_enabled");
    assert_has_param(&template, "project_name");
    assert_has_param(&template, "credit_symbol");
    assert_has_param(&template, "total_credits_tonnes");
    assert_has_param(&template, "verification_body");
    assert_has_param(&template, "project_location");

    // credit_standard should be required
    let standard_param = template
        .deployment_parameters
        .iter()
        .find(|p| p.name == "credit_standard")
        .unwrap();
    assert!(standard_param.required, "credit_standard should be required");

    // vintage_year should be required
    let vintage_param = template
        .deployment_parameters
        .iter()
        .find(|p| p.name == "vintage_year")
        .unwrap();
    assert!(vintage_param.required, "vintage_year should be required");

    Ok(())
}

#[tokio::test]
async fn test_art_collectible_deployment_parameters() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::ArtCollectibleToken)
        .await?;

    assert_has_param(&template, "item_type");
    assert_has_param(&template, "artist_creator");
    assert_has_param(&template, "appraisal_value_usd");
    assert_has_param(&template, "provenance_verified");
    assert_has_param(&template, "item_name");
    assert_has_param(&template, "item_symbol");
    assert_has_param(&template, "creation_year");
    assert_has_param(&template, "total_fractions");
    assert_has_param(&template, "insurance_enabled");
    assert_has_param(&template, "physical_custody");
    assert_has_param(&template, "kyc_required");

    // item_type should be required
    let item_type_param = template
        .deployment_parameters
        .iter()
        .find(|p| p.name == "item_type")
        .unwrap();
    assert!(item_type_param.required, "item_type should be required");

    // artist_creator should be required
    let artist_param = template
        .deployment_parameters
        .iter()
        .find(|p| p.name == "artist_creator")
        .unwrap();
    assert!(artist_param.required, "artist_creator should be required");

    Ok(())
}

#[tokio::test]
async fn test_ip_revenue_deployment_parameters() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::IPRevenueToken)
        .await?;

    assert_has_param(&template, "ip_type");
    assert_has_param(&template, "revenue_share_percent");
    assert_has_param(&template, "jurisdiction");
    assert_has_param(&template, "sublicensing_allowed");
    assert_has_param(&template, "ip_name");
    assert_has_param(&template, "ip_symbol");
    assert_has_param(&template, "registration_number");
    assert_has_param(&template, "expiry_date");
    assert_has_param(&template, "total_tokens");
    assert_has_param(&template, "licensor");

    // ip_type should be required
    let ip_type_param = template
        .deployment_parameters
        .iter()
        .find(|p| p.name == "ip_type")
        .unwrap();
    assert!(ip_type_param.required, "ip_type should be required");

    // jurisdiction should be required
    let jurisdiction_param = template
        .deployment_parameters
        .iter()
        .find(|p| p.name == "jurisdiction")
        .unwrap();
    assert!(
        jurisdiction_param.required,
        "jurisdiction should be required"
    );

    Ok(())
}

#[tokio::test]
async fn test_physical_goods_deployment_parameters() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::PhysicalGoodsToken)
        .await?;

    assert_has_param(&template, "product_category");
    assert_has_param(&template, "manufacturer");
    assert_has_param(&template, "redemption_enabled");
    assert_has_param(&template, "serial_number_tracking");
    assert_has_param(&template, "product_name");
    assert_has_param(&template, "product_symbol");
    assert_has_param(&template, "quantity_per_token");
    assert_has_param(&template, "total_tokens");
    assert_has_param(&template, "warehouse_location");
    assert_has_param(&template, "supply_chain_verified");
    assert_has_param(&template, "shipping_included");
    assert_has_param(&template, "insurance_enabled");

    // product_category should be required
    let category_param = template
        .deployment_parameters
        .iter()
        .find(|p| p.name == "product_category")
        .unwrap();
    assert!(
        category_param.required,
        "product_category should be required"
    );

    // manufacturer should be required
    let manufacturer_param = template
        .deployment_parameters
        .iter()
        .find(|p| p.name == "manufacturer")
        .unwrap();
    assert!(
        manufacturer_param.required,
        "manufacturer should be required"
    );

    Ok(())
}

// ============================================================================
// D) ABI Function Tests
// ============================================================================

#[tokio::test]
async fn test_real_estate_abi_functions() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::RealEstateToken)
        .await?;

    assert_has_abi_function(&template, "transfer");
    assert_has_abi_function(&template, "updateValuation");
    assert_has_abi_function(&template, "distributeDividend");
    assert_has_abi_function(&template, "updateOccupancy");
    assert_has_abi_function(&template, "verifyKYC");
    assert_has_abi_function(&template, "freezeAccount");
    assert_has_abi_function(&template, "getPropertyDetails");

    Ok(())
}

#[tokio::test]
async fn test_equity_abi_functions() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::EquityToken)
        .await?;

    assert_has_abi_function(&template, "transfer");
    assert_has_abi_function(&template, "vote");
    assert_has_abi_function(&template, "distributeDividend");
    assert_has_abi_function(&template, "vest");
    assert_has_abi_function(&template, "lockShares");
    assert_has_abi_function(&template, "unlockShares");
    assert_has_abi_function(&template, "getShareholderInfo");

    Ok(())
}

#[tokio::test]
async fn test_fixed_income_abi_functions() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::FixedIncomeToken)
        .await?;

    assert_has_abi_function(&template, "transfer");
    assert_has_abi_function(&template, "payCoupon");
    assert_has_abi_function(&template, "redeem");
    assert_has_abi_function(&template, "call");
    assert_has_abi_function(&template, "convert");
    assert_has_abi_function(&template, "getCouponSchedule");
    assert_has_abi_function(&template, "getYieldToMaturity");

    Ok(())
}

#[tokio::test]
async fn test_commodity_abi_functions() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::CommodityToken)
        .await?;

    assert_has_abi_function(&template, "transfer");
    assert_has_abi_function(&template, "updateSpotPrice");
    assert_has_abi_function(&template, "requestDelivery");
    assert_has_abi_function(&template, "verifyStorage");
    assert_has_abi_function(&template, "getStorageProof");
    assert_has_abi_function(&template, "redeemPhysical");

    Ok(())
}

#[tokio::test]
async fn test_carbon_credit_abi_functions() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::CarbonCreditToken)
        .await?;

    assert_has_abi_function(&template, "transfer");
    assert_has_abi_function(&template, "retireCredits");
    assert_has_abi_function(&template, "verifyProject");
    assert_has_abi_function(&template, "getRetirementCertificate");
    assert_has_abi_function(&template, "updateVerification");
    assert_has_abi_function(&template, "getProjectImpact");

    Ok(())
}

#[tokio::test]
async fn test_art_collectible_abi_functions() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::ArtCollectibleToken)
        .await?;

    assert_has_abi_function(&template, "transfer");
    assert_has_abi_function(&template, "updateAppraisal");
    assert_has_abi_function(&template, "addProvenance");
    assert_has_abi_function(&template, "verifyAuthenticity");
    assert_has_abi_function(&template, "requestPhysicalTransfer");
    assert_has_abi_function(&template, "getFractionValue");

    Ok(())
}

#[tokio::test]
async fn test_ip_revenue_abi_functions() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::IPRevenueToken)
        .await?;

    assert_has_abi_function(&template, "transfer");
    assert_has_abi_function(&template, "distributeRevenue");
    assert_has_abi_function(&template, "updateRevenueReport");
    assert_has_abi_function(&template, "verifyRegistration");
    assert_has_abi_function(&template, "getLicenseTerms");
    assert_has_abi_function(&template, "getRevenueHistory");

    Ok(())
}

#[tokio::test]
async fn test_physical_goods_abi_functions() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::PhysicalGoodsToken)
        .await?;

    assert_has_abi_function(&template, "transfer");
    assert_has_abi_function(&template, "redeemPhysical");
    assert_has_abi_function(&template, "updateInventory");
    assert_has_abi_function(&template, "verifySupplyChain");
    assert_has_abi_function(&template, "getTrackingInfo");
    assert_has_abi_function(&template, "requestShipping");

    Ok(())
}

#[tokio::test]
async fn test_all_rwa_abis_have_transfer_function() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        assert_has_abi_function(&template, "transfer");
    }

    Ok(())
}

#[tokio::test]
async fn test_all_rwa_abis_have_constructor() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        assert!(
            template.abi.constructor.is_some(),
            "{:?} ABI missing constructor",
            rwa_type
        );
    }

    Ok(())
}

// ============================================================================
// E) Gas Estimation Tests
// ============================================================================

#[tokio::test]
async fn test_all_rwa_templates_have_gas_estimates() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        assert!(
            template.gas_estimates.deployment > 0,
            "{:?} has zero deployment gas estimate",
            rwa_type
        );
        assert!(
            !template.gas_estimates.function_calls.is_empty(),
            "{:?} has no function call gas estimates",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_rwa_gas_estimates_are_reasonable() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    // All RWA deployment gas should be between 3M and 5M
    let min_gas: u64 = 3_000_000;
    let max_gas: u64 = 5_000_000;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        let gas = template.gas_estimates.deployment;
        assert!(
            gas >= min_gas && gas <= max_gas,
            "{:?} deployment gas {} is outside reasonable range [{}, {}]",
            rwa_type,
            gas,
            min_gas,
            max_gas
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_gas_estimator_has_all_rwa_base_costs() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let engine = &ecosystem.deployment_engine;
    let estimator = &engine.gas_estimator;

    for rwa_type in all_rwa_types() {
        assert!(
            estimator.base_costs.contains_key(&rwa_type),
            "{:?} not found in GasEstimator base_costs",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_gas_estimator_base_costs_match_template_deployment_costs() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let estimator = &ecosystem.deployment_engine.gas_estimator;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        let estimator_gas = estimator.base_costs.get(&rwa_type).copied().unwrap_or(0);
        let template_gas = template.gas_estimates.deployment;
        assert_eq!(
            estimator_gas, template_gas,
            "{:?} gas mismatch: GasEstimator={} vs Template={}",
            rwa_type, estimator_gas, template_gas
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_real_estate_specific_gas_estimates() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::RealEstateToken)
        .await?;

    assert_eq!(template.gas_estimates.deployment, 4_500_000);
    assert!(template.gas_estimates.function_calls.contains_key("transfer"));
    assert!(template
        .gas_estimates
        .function_calls
        .contains_key("distributeDividend"));
    assert!(template
        .gas_estimates
        .function_calls
        .contains_key("updateValuation"));

    Ok(())
}

#[tokio::test]
async fn test_carbon_credit_specific_gas_estimates() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    let template = ecosystem
        .get_template(&ContractType::CarbonCreditToken)
        .await?;

    assert_eq!(template.gas_estimates.deployment, 3_800_000);
    assert!(template
        .gas_estimates
        .function_calls
        .contains_key("retireCredits"));
    assert!(template
        .gas_estimates
        .function_calls
        .contains_key("verifyProject"));

    Ok(())
}

// ============================================================================
// F) Contract Type Enum Tests
// ============================================================================

#[test]
fn test_all_8_rwa_types_exist_in_contract_type_enum() {
    // This test verifies that the ContractType enum contains all 8 RWA variants.
    // If any variant is removed or renamed, this test will fail to compile.
    let types: Vec<ContractType> = vec![
        ContractType::RealEstateToken,
        ContractType::CommodityToken,
        ContractType::CarbonCreditToken,
        ContractType::EquityToken,
        ContractType::FixedIncomeToken,
        ContractType::IPRevenueToken,
        ContractType::PhysicalGoodsToken,
        ContractType::ArtCollectibleToken,
    ];
    assert_eq!(types.len(), 8);
}

#[test]
fn test_rwa_types_are_distinct_from_each_other() {
    let types = all_rwa_types();

    // Verify all pairs are distinct via PartialEq
    for i in 0..types.len() {
        for j in (i + 1)..types.len() {
            assert_ne!(
                types[i], types[j],
                "RWA types at index {} and {} should be distinct",
                i, j
            );
        }
    }
}

#[test]
fn test_rwa_types_are_distinct_from_non_rwa_types() {
    let rwa_types = all_rwa_types();
    let non_rwa_types = vec![
        ContractType::SecureToken,
        ContractType::AdvancedToken,
        ContractType::RwaToken,
        ContractType::MultisigWallet,
        ContractType::Governance,
        ContractType::PrivateDex,
    ];

    for rwa in &rwa_types {
        for non_rwa in &non_rwa_types {
            assert_ne!(
                rwa, non_rwa,
                "{:?} should not equal {:?}",
                rwa, non_rwa
            );
        }
    }
}

#[test]
fn test_rwa_contract_type_serialization_roundtrip() {
    for rwa_type in all_rwa_types() {
        let serialized = serde_json::to_string(&rwa_type)
            .unwrap_or_else(|e| panic!("Failed to serialize {:?}: {}", rwa_type, e));
        let deserialized: ContractType = serde_json::from_str(&serialized)
            .unwrap_or_else(|e| panic!("Failed to deserialize {:?} from '{}': {}", rwa_type, serialized, e));
        assert_eq!(
            rwa_type, deserialized,
            "Serialization roundtrip failed for {:?}",
            rwa_type
        );
    }
}

#[test]
fn test_rwa_contract_type_debug_format() {
    // Ensure Debug is implemented and produces meaningful output
    for rwa_type in all_rwa_types() {
        let debug_str = format!("{:?}", rwa_type);
        assert!(
            !debug_str.is_empty(),
            "Debug format should not be empty for {:?}",
            rwa_type
        );
    }
}

#[test]
fn test_rwa_contract_type_clone() {
    for rwa_type in all_rwa_types() {
        let cloned = rwa_type.clone();
        assert_eq!(rwa_type, cloned, "Clone should produce equal value");
    }
}

#[test]
fn test_rwa_contract_type_hash_uniqueness() {
    use std::collections::HashSet;

    let rwa_types = all_rwa_types();
    let unique: HashSet<ContractType> = rwa_types.iter().cloned().collect();

    assert_eq!(
        unique.len(),
        8,
        "All 8 RWA types should have unique hashes, got {} unique",
        unique.len()
    );
}

// ============================================================================
// Additional Cross-Cutting Validation Tests
// ============================================================================

#[tokio::test]
async fn test_form_definition_schemas_are_valid_json_objects() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let form = ecosystem.get_form_definition(&rwa_type).await?;

        assert!(
            form.form_schema.is_object(),
            "{:?} form_schema is not a JSON object",
            rwa_type
        );
        assert!(
            form.validation_schema.is_object(),
            "{:?} validation_schema is not a JSON object",
            rwa_type
        );
        assert!(
            form.ui_schema.is_object(),
            "{:?} ui_schema is not a JSON object",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_form_definition_contract_types_match() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let form = ecosystem.get_form_definition(&rwa_type).await?;
        assert_eq!(
            form.contract_type, rwa_type,
            "FormDefinition contract_type does not match for {:?}",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_form_schemas_have_required_fields_array() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let form = ecosystem.get_form_definition(&rwa_type).await?;
        let required = form.form_schema.get("required");
        assert!(
            required.is_some(),
            "{:?} form_schema is missing 'required' array",
            rwa_type
        );
        assert!(
            required.unwrap().is_array(),
            "{:?} form_schema 'required' should be an array",
            rwa_type
        );
        assert!(
            !required.unwrap().as_array().unwrap().is_empty(),
            "{:?} form_schema 'required' array should not be empty",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_form_schemas_have_properties_section() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let form = ecosystem.get_form_definition(&rwa_type).await?;
        let properties = form.form_schema.get("properties");
        assert!(
            properties.is_some(),
            "{:?} form_schema is missing 'properties'",
            rwa_type
        );
        assert!(
            properties.unwrap().is_object(),
            "{:?} form_schema 'properties' should be an object",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_all_rwa_templates_have_cost_estimate() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        let cost = &template.form_config.cost_estimate;
        assert!(
            cost.gas_cost > 0,
            "{:?} cost_estimate.gas_cost is zero",
            rwa_type
        );
        assert!(
            !cost.total_cost_orb.is_empty(),
            "{:?} cost_estimate.total_cost_orb is empty",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_cost_estimate_gas_matches_deployment_gas() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        assert_eq!(
            template.form_config.cost_estimate.gas_cost,
            template.gas_estimates.deployment,
            "{:?} cost_estimate.gas_cost does not match gas_estimates.deployment",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_all_rwa_templates_have_feature_costs() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        assert!(
            !template.gas_estimates.feature_costs.is_empty(),
            "{:?} has no feature_costs in gas_estimates",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_all_rwa_abi_functions_have_gas_estimates() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        for func in &template.abi.functions {
            assert!(
                func.gas_estimate.is_some(),
                "{:?} ABI function '{}' missing gas_estimate",
                rwa_type,
                func.name
            );
            assert!(
                func.gas_estimate.unwrap() > 0,
                "{:?} ABI function '{}' has zero gas_estimate",
                rwa_type,
                func.name
            );
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_all_rwa_templates_have_events() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        assert!(
            !template.abi.events.is_empty(),
            "{:?} ABI has no events",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_all_rwa_templates_have_faq() -> Result<()> {
    let ecosystem = create_ecosystem().await;

    for rwa_type in all_rwa_types() {
        let template = ecosystem.get_template(&rwa_type).await?;
        assert!(
            !template.documentation.faq.is_empty(),
            "{:?} has no FAQ entries",
            rwa_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_ecosystem_get_template_returns_error_for_nonexistent() -> Result<()> {
    let ecosystem = create_ecosystem().await;
    // OptionsContract and PredictionMarket load as empty (Ok(())), so they should NOT
    // have templates registered. This tests that get_template returns an error
    // for an unregistered type.
    let result = ecosystem
        .get_template(&ContractType::OptionsContract)
        .await;
    assert!(
        result.is_err(),
        "get_template should return error for unregistered OptionsContract"
    );

    Ok(())
}
