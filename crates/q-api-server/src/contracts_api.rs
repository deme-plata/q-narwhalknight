/// Smart Contract API Endpoints for Q-NarwhalKnight
///
/// This module provides REST API endpoints for deploying and managing
/// Orobit Chimera smart contracts through frontend forms.
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::AppState;
use q_vm::contracts::{
    ContractAddress, ContractType, DeployedSmartContract, DeploymentOptions, FormDefinition,
    OrobitSmartContractEcosystem, SmartContractTemplate,
};

/// API response wrapper
#[derive(Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: u64,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: current_timestamp(),
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message),
            timestamp: current_timestamp(),
        }
    }
}

/// Contract deployment request from frontend
#[derive(Debug, Deserialize)]
pub struct DeploymentRequest {
    pub contract_type: String, // Will be parsed to ContractType
    pub owner: String,         // Hex-encoded address
    pub parameters: HashMap<String, serde_json::Value>,
    pub deployment_options: Option<FrontendDeploymentOptions>,
}

#[derive(Debug, Deserialize)]
pub struct FrontendDeploymentOptions {
    pub test_deployment: Option<bool>,
    pub auto_verify: Option<bool>,
    pub enable_governance: Option<bool>,
    pub enable_upgrades: Option<bool>,
    pub gas_limit: Option<u64>,
    pub deploy_with_proxy: Option<bool>,
}

/// Contract information for frontend display
#[derive(Debug, Serialize)]
pub struct ContractInfo {
    pub address: String,
    pub contract_type: String,
    pub name: String,
    pub symbol: Option<String>,
    pub owner: String,
    pub deployed_at: u64,
    pub verified: bool,
    pub has_security_features: bool,
    pub features: HashMap<String, bool>,
    pub deployment_tx: String,
}

/// Deployment status response
#[derive(Debug, Serialize)]
pub struct DeploymentStatusResponse {
    pub request_id: String,
    pub status: String,
    pub contract_address: Option<String>,
    pub deployment_tx: Option<String>,
    pub gas_used: Option<u64>,
    pub error_message: Option<String>,
    pub progress: DeploymentProgress,
}

#[derive(Debug, Serialize)]
pub struct DeploymentProgress {
    pub current_step: u32,
    pub total_steps: u32,
    pub step_name: String,
    pub estimated_time_remaining: u32, // seconds
}

/// Form schema response for frontend
#[derive(Debug, Serialize)]
pub struct FormSchemaResponse {
    pub contract_type: String,
    pub form_title: String,
    pub form_description: String,
    pub schema: serde_json::Value,
    pub ui_schema: serde_json::Value,
    pub validation_schema: serde_json::Value,
    pub examples: Vec<FormExampleResponse>,
    pub gas_estimate: GasEstimateResponse,
}

#[derive(Debug, Serialize)]
pub struct FormExampleResponse {
    pub name: String,
    pub description: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct GasEstimateResponse {
    pub base_gas: u64,
    pub total_gas_estimate: u64,
    pub gas_price_gwei: u64,
    pub estimated_cost_orb: String,
    pub estimated_cost_usd: Option<String>,
}

/// Contract templates list response
#[derive(Debug, Serialize)]
pub struct TemplatesListResponse {
    pub templates: Vec<TemplateInfo>,
    pub categories: HashMap<String, Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct TemplateInfo {
    pub contract_type: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub category: String,
    pub complexity: String, // "beginner", "intermediate", "advanced"
    pub gas_estimate: u64,
    pub features: Vec<String>,
    pub security_level: String,
    pub audit_status: String,
}

/// Query parameters for filtering contracts
#[derive(Debug, Deserialize)]
pub struct ContractQuery {
    pub owner: Option<String>,
    pub contract_type: Option<String>,
    pub verified_only: Option<bool>,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

/// Create the contracts API router
pub fn create_contracts_router() -> Router<Arc<AppState>> {
    Router::new()
        // Template and form endpoints
        .route("/templates", get(get_contract_templates))
        .route("/templates/:contract_type/form", get(get_deployment_form))
        .route(
            "/templates/:contract_type/estimate",
            post(estimate_deployment_cost),
        )
        // Deployment endpoints
        .route("/deploy", post(deploy_contract))
        .route(
            "/deployments/:request_id/status",
            get(get_deployment_status),
        )
        .route("/deployments", get(get_user_deployments))
        // Contract management endpoints
        .route("/contracts", get(get_contracts))
        .route("/contracts/:address", get(get_contract_details))
        .route("/contracts/:address/interact", post(interact_with_contract))
        // User-specific endpoints
        .route("/user/:address/contracts", get(get_user_contracts))
        .route(
            "/user/:address/deployments",
            get(get_user_deployments),
        )
}

/// Get all available contract templates
pub async fn get_contract_templates(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<TemplatesListResponse>>, StatusCode> {
    let ecosystem = &state.orobit_ecosystem;

    let contract_types = ecosystem.get_available_contracts().await;
    let mut templates = Vec::new();
    let mut categories: HashMap<String, Vec<String>> = HashMap::new();

    for contract_type in contract_types {
        match ecosystem.get_template(&contract_type).await {
            Ok(template) => {
                let category = format!("{:?}", contract_type); // Simplified
                categories
                    .entry(category.clone())
                    .or_insert_with(Vec::new)
                    .push(format!("{:?}", contract_type));

                templates.push(TemplateInfo {
                    contract_type: format!("{:?}", contract_type),
                    name: template.name,
                    description: template.description,
                    version: template.version,
                    category,
                    complexity: if template.deployment_parameters.len() > 5 {
                        "advanced".to_string()
                    } else {
                        "beginner".to_string()
                    },
                    gas_estimate: template.gas_estimates.deployment,
                    features: template
                        .deployment_parameters
                        .iter()
                        .filter(|p| p.param_type == "bool")
                        .map(|p| p.name.clone())
                        .collect(),
                    security_level: if template.security_features.reentrancy_protection
                        && template.security_features.overflow_protection
                    {
                        "high".to_string()
                    } else {
                        "medium".to_string()
                    },
                    audit_status: format!("{:?}", template.security_features.audit_status),
                });
            }
            Err(_) => continue,
        }
    }

    Ok(Json(ApiResponse::success(TemplatesListResponse {
        templates,
        categories,
    })))
}

/// Get deployment form for specific contract type
pub async fn get_deployment_form(
    Path(contract_type_str): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<FormSchemaResponse>>, StatusCode> {
    let contract_type = match parse_contract_type(&contract_type_str) {
        Ok(ct) => ct,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    let ecosystem = &state.orobit_ecosystem;

    match ecosystem.get_form_definition(&contract_type).await {
        Ok(form_def) => match ecosystem.get_template(&contract_type).await {
            Ok(template) => Ok(Json(ApiResponse::success(FormSchemaResponse {
                contract_type: contract_type_str,
                form_title: template.form_config.title,
                form_description: template.form_config.description,
                schema: form_def.form_schema,
                ui_schema: form_def.ui_schema,
                validation_schema: form_def.validation_schema,
                examples: form_def
                    .examples
                    .into_iter()
                    .map(|ex| FormExampleResponse {
                        name: ex.name,
                        description: ex.description,
                        data: ex.data,
                    })
                    .collect(),
                gas_estimate: GasEstimateResponse {
                    base_gas: template.gas_estimates.deployment,
                    total_gas_estimate: template.gas_estimates.deployment,
                    gas_price_gwei: 1,
                    estimated_cost_orb: template.form_config.cost_estimate.total_cost_orb,
                    estimated_cost_usd: template.form_config.cost_estimate.usd_equivalent,
                },
            }))),
            Err(e) => Ok(Json(ApiResponse::error(format!(
                "Template not found: {}",
                e
            )))),
        },
        Err(e) => Ok(Json(ApiResponse::error(format!(
            "Form definition not found: {}",
            e
        )))),
    }
}

/// Deploy a contract from frontend form
pub async fn deploy_contract(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DeploymentRequest>,
) -> Result<Json<ApiResponse<DeploymentStatusResponse>>, StatusCode> {
    // Parse contract type
    let contract_type = match parse_contract_type(&request.contract_type) {
        Ok(ct) => ct,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    // Parse deployer address
    let deployer = match parse_address(&request.owner) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    // Convert deployment options
    let deployment_options = request
        .deployment_options
        .map(|opts| DeploymentOptions {
            test_deployment: opts.test_deployment.unwrap_or(false),
            auto_verify: opts.auto_verify.unwrap_or(false),
            enable_governance: opts.enable_governance.unwrap_or(false),
            enable_upgrades: opts.enable_upgrades.unwrap_or(false),
            gas_limit: opts.gas_limit,
            deploy_with_proxy: opts.deploy_with_proxy.unwrap_or(false),
        })
        .unwrap_or(DeploymentOptions {
            test_deployment: false,
            auto_verify: false,
            enable_governance: false,
            enable_upgrades: false,
            gas_limit: None,
            deploy_with_proxy: false,
        });

    let ecosystem = &state.orobit_ecosystem;

    match ecosystem
        .deploy_contract(
            contract_type,
            deployer,
            request.parameters,
            deployment_options,
        )
        .await
    {
        Ok(request_id) => Ok(Json(ApiResponse::success(DeploymentStatusResponse {
            request_id: request_id.clone(),
            status: "pending".to_string(),
            contract_address: None,
            deployment_tx: None,
            gas_used: None,
            error_message: None,
            progress: DeploymentProgress {
                current_step: 1,
                total_steps: 4,
                step_name: "Validation".to_string(),
                estimated_time_remaining: 60,
            },
        }))),
        Err(e) => Ok(Json(ApiResponse::error(format!(
            "Deployment failed: {}",
            e
        )))),
    }
}

/// Get deployment status
pub async fn get_deployment_status(
    Path(request_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<DeploymentStatusResponse>>, StatusCode> {
    // For now, return a mock successful deployment
    // In production, this would query the actual deployment status
    Ok(Json(ApiResponse::success(DeploymentStatusResponse {
        request_id: request_id.clone(),
        status: "deployed".to_string(),
        contract_address: Some("0x1234567890123456789012345678901234567890".to_string()),
        deployment_tx: Some("0xabcdef1234567890abcdef1234567890abcdef12".to_string()),
        gas_used: Some(2_500_000),
        error_message: None,
        progress: DeploymentProgress {
            current_step: 4,
            total_steps: 4,
            step_name: "Completed".to_string(),
            estimated_time_remaining: 0,
        },
    })))
}

/// Get user's contracts
pub async fn get_user_contracts(
    Path(address): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<ContractInfo>>>, StatusCode> {
    let deployer = match parse_address(&address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    let ecosystem = &state.orobit_ecosystem;
    let contracts = ecosystem.get_user_contracts(deployer).await;

    let contract_infos: Vec<ContractInfo> = contracts
        .into_iter()
        .map(|contract| {
            ContractInfo {
                address: hex::encode(contract.address.0),
                contract_type: format!("{:?}", contract.contract_type),
                name: contract.metadata.name,
                symbol: contract.metadata.symbol,
                owner: hex::encode(contract.deployer),
                deployed_at: contract.deployed_at,
                verified: contract.verified,
                has_security_features: true, // From template security features
                features: contract.metadata.features,
                deployment_tx: contract.deployment_tx,
            }
        })
        .collect();

    Ok(Json(ApiResponse::success(contract_infos)))
}

/// Get all contracts with optional filtering
pub async fn get_contracts(
    Query(query): Query<ContractQuery>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<ContractInfo>>>, StatusCode> {
    // Implementation would filter based on query parameters
    // For now, return empty list
    Ok(Json(ApiResponse::success(Vec::new())))
}

/// Get specific contract details
pub async fn get_contract_details(
    Path(address): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<ContractInfo>>, StatusCode> {
    // Implementation would look up specific contract
    // For now, return error
    Ok(Json(ApiResponse::error("Contract not found".to_string())))
}

/// Interact with deployed contract
pub async fn interact_with_contract(
    Path(address): Path<String>,
    State(state): State<Arc<AppState>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    // Implementation would execute contract function
    // For now, return success
    Ok(Json(ApiResponse::success(serde_json::json!({
        "result": "success",
        "transaction_hash": "0xmockresult123456789"
    }))))
}

/// Get user's deployment history
pub async fn get_user_deployments(
    Path(address): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<DeploymentStatusResponse>>>, StatusCode> {
    // Implementation would return deployment history
    // For now, return empty list
    Ok(Json(ApiResponse::success(Vec::new())))
}

/// Estimate deployment cost
pub async fn estimate_deployment_cost(
    Path(contract_type_str): Path<String>,
    State(state): State<Arc<AppState>>,
    Json(parameters): Json<HashMap<String, serde_json::Value>>,
) -> Result<Json<ApiResponse<GasEstimateResponse>>, StatusCode> {
    let contract_type = match parse_contract_type(&contract_type_str) {
        Ok(ct) => ct,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    let ecosystem = &state.orobit_ecosystem;

    match ecosystem.get_template(&contract_type).await {
        Ok(template) => {
            // Calculate gas estimate based on enabled features
            let mut total_gas = template.gas_estimates.deployment;

            for (param_name, value) in &parameters {
                if value.as_bool().unwrap_or(false) {
                    if let Some(feature_cost) = template.gas_estimates.feature_costs.get(param_name)
                    {
                        total_gas += feature_cost;
                    }
                }
            }

            Ok(Json(ApiResponse::success(GasEstimateResponse {
                base_gas: template.gas_estimates.deployment,
                total_gas_estimate: total_gas,
                gas_price_gwei: 1,
                estimated_cost_orb: format!("{:.6}", (total_gas as f64) * 0.000000001),
                estimated_cost_usd: Some(format!(
                    "${:.2}",
                    (total_gas as f64) * 0.000000001 * 1000.0
                )),
            })))
        }
        Err(e) => Ok(Json(ApiResponse::error(format!(
            "Template not found: {}",
            e
        )))),
    }
}

// Helper functions
fn parse_contract_type(contract_type_str: &str) -> Result<ContractType, String> {
    match contract_type_str.to_lowercase().as_str() {
        "secure_token" => Ok(ContractType::SecureToken),
        "advanced_token" => Ok(ContractType::AdvancedToken),
        "rwa_token" => Ok(ContractType::RwaToken),
        "orbusd_stablecoin" => Ok(ContractType::OrbusdStablecoin),
        "multisig_wallet" => Ok(ContractType::MultisigWallet),
        "governance" => Ok(ContractType::Governance),
        "private_dex" => Ok(ContractType::PrivateDex),
        "timelock_vault" => Ok(ContractType::TimelockVault),
        "oracle_feed" => Ok(ContractType::OracleFeed),
        _ => Err(format!("Unknown contract type: {}", contract_type_str)),
    }
}

fn parse_address(address_str: &str) -> Result<[u8; 32], String> {
    if address_str.len() != 42 || !address_str.starts_with("0x") {
        return Err("Invalid address format".to_string());
    }

    match hex::decode(&address_str[2..]) {
        Ok(bytes) => {
            if bytes.len() == 20 {
                // Ethereum-style address (20 bytes), pad to 32 bytes
                let mut padded = [0u8; 32];
                padded[12..].copy_from_slice(&bytes);
                Ok(padded)
            } else {
                Err("Address must be 20 bytes".to_string())
            }
        }
        Err(_) => Err("Invalid hex in address".to_string()),
    }
}


fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
