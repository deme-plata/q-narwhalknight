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
use q_types::{Transaction, TxStatus};
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
    pub total_supply: Option<u64>, // Add total supply for tokens
    pub decimals: Option<u32>,     // Add decimals for display
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
        .route("/deployed", get(get_contracts))
        .route("/:address", get(get_contract_details))
        .route("/:address/interact", post(interact_with_contract))
        .route(
            "/:token_address/balance/:wallet_address",
            get(get_token_balance),
        )
        // Token operations endpoints
        .route("/mint", post(mint_tokens))
        .route("/burn", post(burn_tokens))
        .route("/airdrop", post(airdrop_tokens))
        .route("/pause", post(pause_contract))
        .route("/reflection", post(update_reflection_rate))
        // User-specific endpoints
        .route("/user/:address/contracts", get(get_user_contracts))
        .route("/user/:address/deployments", get(get_user_deployments))
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
            request.parameters.clone(),
            deployment_options,
        )
        .await
    {
        Ok((request_id, contract_address)) => {
            // Deduct deployment cost from deployer's native QUG balance
            const DEPLOYMENT_COST: u64 = 100_000_000; // 1 QUG = 100M smallest units
            {
                let mut wallet_balances = state.wallet_balances.write().await;
                if let Some(balance) = wallet_balances.get_mut(&deployer) {
                    if *balance >= DEPLOYMENT_COST {
                        *balance -= DEPLOYMENT_COST;
                        tracing::info!(
                            "💸 Deducted {} QUG deployment cost from {}. New balance: {}",
                            DEPLOYMENT_COST as f64 / 100_000_000.0,
                            hex::encode(deployer),
                            *balance as f64 / 100_000_000.0
                        );

                        // Create transaction history entry for deployment
                        let tx_hash = format!(
                            "deploy-{}-{}",
                            hex::encode(contract_address.0),
                            chrono::Utc::now().timestamp_millis()
                        );
                        let transaction = Transaction {
                            id: [0u8; 32], // Would be properly hashed in production
                            from: deployer,
                            to: contract_address.0,
                            amount: DEPLOYMENT_COST,
                            fee: 0,
                            nonce: 0,
                            signature: vec![],
                            timestamp: chrono::Utc::now(),
                            data: format!("Contract deployment: {}", request.contract_type)
                                .into_bytes(),
                            token_type: q_types::TokenType::QUG,
                            fee_token_type: q_types::TokenType::QUGUSD,
                        };
                        // Store in transaction pool for history
                        let tx_id = transaction.id;
                        state.tx_pool.insert(tx_id, transaction);
                        state.tx_status.insert(
                            tx_id,
                            TxStatus::Confirmed {
                                block_height: 0,
                                round: 0,
                            },
                        );
                    } else {
                        tracing::warn!(
                            "⚠️ Insufficient balance for deployment. Required: {}, Available: {}",
                            DEPLOYMENT_COST,
                            *balance
                        );
                    }
                } else {
                    tracing::warn!("⚠️ Deployer wallet not found: {}", hex::encode(deployer));
                }
            }

            // Mint initial supply to deployer if this is a token contract
            if let Some(initial_supply_val) = request
                .parameters
                .get("initialSupply")
                .or_else(|| request.parameters.get("initial_supply"))
            {
                // Get decimals from parameters (for display purposes only)
                let decimals = request
                    .parameters
                    .get("decimals")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(18) as u32;

                // ULTRA-SIMPLE APPROACH: Number you enter = Number you get (in base units)
                // User enters 1 → Gets 1 base unit
                // User enters 1000000000 → Gets 1 billion base units
                // User enters 10000000000000000000000 → Gets 10 sextillion base units
                //
                // NO multiplication or division - just use the raw number!

                let initial_supply_result: Option<u64> = if let Some(supply_u64) =
                    initial_supply_val.as_u64()
                {
                    // Number fits in u64 - use it directly
                    Some(supply_u64)
                } else if let Some(supply_str) = initial_supply_val.as_str() {
                    // Large number as string
                    match supply_str.parse::<u128>() {
                        Ok(supply_u128) => {
                            if supply_u128 <= u64::MAX as u128 {
                                let result = supply_u128 as u64;
                                tracing::info!(
                                    "✅ User entered {} base units → stored as {} base units",
                                    supply_str,
                                    result
                                );
                                Some(result)
                            } else {
                                tracing::error!(
                                    "❌ Initial supply {} exceeds u64::MAX ({}). Maximum allowed: {}",
                                    supply_str,
                                    supply_u128,
                                    u64::MAX
                                );
                                None
                            }
                        }
                        Err(_) => {
                            tracing::warn!(
                                "⚠️ Could not parse initial supply string: {}",
                                supply_str
                            );
                            None
                        }
                    }
                } else {
                    None
                };

                match initial_supply_result {
                    Some(initial_supply) if initial_supply > 0 => {
                        // Mint tokens to deployer's wallet
                        let mut token_balances = state.token_balances.write().await;
                        token_balances.insert((deployer, contract_address.0), initial_supply);

                        // Calculate human-readable amount (for logging only)
                        let token_amount = initial_supply as f64 / 10f64.powi(decimals as i32);
                        tracing::info!(
                            "💰 Minted {} base units (displayed as {} tokens with {} decimals) to deployer {}",
                            initial_supply,
                            token_amount,
                            decimals,
                            hex::encode(deployer)
                        );

                        // Persist token balance to storage
                        drop(token_balances); // Release write lock before async operation
                        if let Err(e) = state
                            .storage_engine
                            .save_token_balance(&deployer, &contract_address.0, initial_supply)
                            .await
                        {
                            tracing::warn!("Failed to persist token balance: {}", e);
                        }
                    }
                    None => {
                        tracing::warn!("⚠️ Initial supply could not be processed - either zero or invalid format");
                    }
                    _ => {
                        // initial_supply is 0, skip minting
                        tracing::debug!("Initial supply is 0, skipping minting");
                    }
                }
            }

            // Format contract address with qnk prefix
            let formatted_address = format!("qnk{}", hex::encode(contract_address.0));

            Ok(Json(ApiResponse::success(DeploymentStatusResponse {
                request_id: request_id.clone(),
                status: "deployed".to_string(),
                contract_address: Some(formatted_address),
                deployment_tx: Some(request_id.clone()),
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
            // Extract total_supply and decimals from deployment_params
            // Note: The parameter is stored as "initial_supply" in deployment_params
            let total_supply = contract
                .deployment_params
                .get("initial_supply")
                .and_then(|v| {
                    // Handle both number and string formats
                    v.as_u64()
                        .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
                });

            let decimals = contract
                .deployment_params
                .get("decimals")
                .and_then(|v| v.as_u64())
                .map(|d| d as u32)
                .or(Some(18)); // Default to 18 decimals if not specified

            ContractInfo {
                address: format!("qnk{}", hex::encode(contract.address.0)), // Add qnk prefix to match wallet format
                contract_type: format!("{:?}", contract.contract_type),
                name: contract.metadata.name,
                symbol: contract.metadata.symbol,
                owner: format!("qnk{}", hex::encode(contract.deployer)), // Add qnk prefix
                deployed_at: contract.deployed_at,
                verified: contract.verified,
                has_security_features: true, // From template security features
                features: contract.metadata.features,
                deployment_tx: contract.deployment_tx,
                total_supply,
                decimals,
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
    // Parse the contract address
    let contract_addr = match parse_address(&address) {
        Ok(addr) => ContractAddress(addr),
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    let ecosystem = &state.orobit_ecosystem;

    // Fetch the contract from the deployed contracts map
    match ecosystem.get_contract_by_address(contract_addr).await {
        Some(contract) => {
            // Extract total_supply and decimals from deployment_params
            // Note: The parameter is stored as "initial_supply" in deployment_params
            let total_supply = contract
                .deployment_params
                .get("initial_supply")
                .and_then(|v| {
                    // Handle both number and string formats
                    v.as_u64()
                        .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
                });

            let decimals = contract
                .deployment_params
                .get("decimals")
                .and_then(|v| v.as_u64())
                .map(|d| d as u32)
                .or(Some(18)); // Default to 18 decimals if not specified

            let contract_info = ContractInfo {
                address: format!("qnk{}", hex::encode(contract.address.0)),
                contract_type: format!("{:?}", contract.contract_type),
                name: contract.metadata.name.clone(),
                symbol: contract.metadata.symbol.clone(),
                owner: format!("qnk{}", hex::encode(contract.deployer)),
                deployed_at: contract.deployed_at,
                verified: contract.verified,
                has_security_features: true,
                features: contract.metadata.features.clone(),
                deployment_tx: contract.deployment_tx.clone(),
                total_supply,
                decimals,
            };
            Ok(Json(ApiResponse::success(contract_info)))
        }
        None => Ok(Json(ApiResponse::error(format!(
            "Contract not found at address: {}",
            address
        )))),
    }
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

/// Token balance response
#[derive(Debug, Serialize)]
pub struct TokenBalanceResponse {
    pub balance: u64,
}

/// Get token balance for a wallet
pub async fn get_token_balance(
    Path((token_address, wallet_address)): Path<(String, String)>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<TokenBalanceResponse>>, StatusCode> {
    // Parse addresses
    let token_addr = match parse_address(&token_address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    let wallet_addr = match parse_address(&wallet_address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    // First try: Get balance from in-memory token_balances map
    let balance = {
        let token_balances = state.token_balances.read().await;
        token_balances.get(&(wallet_addr, token_addr)).copied()
    };

    // If not found in memory, try loading from storage and update memory
    let balance = match balance {
        Some(bal) => bal,
        None => {
            // Try loading from persistent storage
            match state
                .storage_engine
                .get_token_balance(&wallet_addr, &token_addr)
                .await
            {
                Ok(stored_balance) => {
                    // Update in-memory cache
                    let mut token_balances = state.token_balances.write().await;
                    token_balances.insert((wallet_addr, token_addr), stored_balance);
                    tracing::debug!(
                        "💾 Loaded token balance from storage: wallet={}, token={}, balance={}",
                        hex::encode(wallet_addr),
                        hex::encode(token_addr),
                        stored_balance
                    );
                    stored_balance
                }
                Err(_) => {
                    // Not found in storage either, return 0
                    0
                }
            }
        }
    };

    tracing::debug!(
        "🔍 Token balance query: wallet={}, token={}, balance={}",
        hex::encode(wallet_addr),
        hex::encode(token_addr),
        balance
    );

    Ok(Json(ApiResponse::success(TokenBalanceResponse { balance })))
}

fn parse_address(address_str: &str) -> Result<[u8; 32], String> {
    // Support both 0x (Ethereum-style) and qnk (Q-NarwhalKnight) prefixes
    let hex_str = if address_str.starts_with("0x") {
        if address_str.len() != 42 && address_str.len() != 66 {
            return Err(format!(
                "Invalid 0x address format (expected 42 or 66 chars, got {})",
                address_str.len()
            ));
        }
        &address_str[2..]
    } else if address_str.starts_with("qnk") {
        // Q-NarwhalKnight addresses: qnk + 40 hex chars = 43 total OR qnk + 64 hex chars = 67 total
        if address_str.len() != 43 && address_str.len() != 67 {
            return Err(format!(
                "Invalid qnk address format (expected 43 or 67 chars, got {})",
                address_str.len()
            ));
        }
        &address_str[3..]
    } else {
        return Err(format!(
            "Address must start with 0x or qnk (got: {})",
            address_str
        ));
    };

    match hex::decode(hex_str) {
        Ok(bytes) => {
            if bytes.len() == 32 {
                // Q-NarwhalKnight native format (32 bytes)
                let mut result = [0u8; 32];
                result.copy_from_slice(&bytes);
                Ok(result)
            } else if bytes.len() == 20 {
                // Ethereum-style address (20 bytes), pad to 32 bytes
                let mut padded = [0u8; 32];
                padded[12..].copy_from_slice(&bytes);
                Ok(padded)
            } else {
                Err(format!(
                    "Address must be 20 or 32 bytes, got {}",
                    bytes.len()
                ))
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

/// Request body for minting tokens
#[derive(Debug, Deserialize)]
pub struct MintRequest {
    pub contract_address: String,
    pub amount: String,
}

/// Request body for burning tokens
#[derive(Debug, Deserialize)]
pub struct BurnRequest {
    pub contract_address: String,
    pub amount: String,
}

/// Request body for airdropping tokens
#[derive(Debug, Deserialize)]
pub struct AirdropRequest {
    pub contract_address: String,
    pub recipients: Vec<String>,
    pub amount_per_recipient: String,
}

/// Response for token operations
#[derive(Debug, Serialize)]
pub struct TokenOperationResponse {
    pub success: bool,
    pub transaction_hash: String,
    pub amount: u64,
    pub message: String,
}

/// Mint tokens to the contract owner
pub async fn mint_tokens(
    State(state): State<Arc<AppState>>,
    Json(request): Json<MintRequest>,
) -> Result<Json<ApiResponse<TokenOperationResponse>>, StatusCode> {
    // Parse contract address
    let contract_addr = match parse_address(&request.contract_address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    // Parse amount
    let amount = match request.amount.parse::<u64>() {
        Ok(amt) if amt > 0 => amt,
        Ok(_) => {
            return Ok(Json(ApiResponse::error(
                "Amount must be greater than 0".to_string(),
            )))
        }
        Err(_) => {
            return Ok(Json(ApiResponse::error(
                "Invalid amount format".to_string(),
            )))
        }
    };

    // Get contract details to verify it exists and has mintable feature
    let ecosystem = &state.orobit_ecosystem;
    let contract = match ecosystem
        .get_contract_by_address(ContractAddress(contract_addr))
        .await
    {
        Some(c) => c,
        None => return Ok(Json(ApiResponse::error("Contract not found".to_string()))),
    };

    // Check if contract has mintable feature
    if !contract
        .metadata
        .features
        .get("mintable")
        .copied()
        .unwrap_or(false)
    {
        return Ok(Json(ApiResponse::error(
            "Contract does not support minting".to_string(),
        )));
    }

    // Mint tokens to the contract owner
    let owner = contract.deployer;
    let new_balance = {
        let mut token_balances = state.token_balances.write().await;
        let current_balance = token_balances
            .get(&(owner, contract_addr))
            .copied()
            .unwrap_or(0);
        let new_balance = current_balance.saturating_add(amount);
        token_balances.insert((owner, contract_addr), new_balance);

        tracing::info!(
            "🪙 Minted {} tokens for contract {} to owner {}. New balance: {}",
            amount,
            hex::encode(contract_addr),
            hex::encode(owner),
            new_balance
        );
        new_balance
    };

    // Persist token balance to storage
    if let Err(e) = state
        .storage_engine
        .save_token_balance(&owner, &contract_addr, new_balance)
        .await
    {
        tracing::warn!("Failed to persist token balance after mint: {}", e);
    }

    // Create transaction hash for the mint operation
    let tx_hash = format!(
        "mint-{}-{}",
        hex::encode(contract_addr),
        chrono::Utc::now().timestamp_millis()
    );

    Ok(Json(ApiResponse::success(TokenOperationResponse {
        success: true,
        transaction_hash: tx_hash,
        amount,
        message: format!("Successfully minted {} tokens", amount),
    })))
}

/// Burn tokens from the contract owner
pub async fn burn_tokens(
    State(state): State<Arc<AppState>>,
    Json(request): Json<BurnRequest>,
) -> Result<Json<ApiResponse<TokenOperationResponse>>, StatusCode> {
    // Parse contract address
    let contract_addr = match parse_address(&request.contract_address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    // Parse amount
    let amount = match request.amount.parse::<u64>() {
        Ok(amt) if amt > 0 => amt,
        Ok(_) => {
            return Ok(Json(ApiResponse::error(
                "Amount must be greater than 0".to_string(),
            )))
        }
        Err(_) => {
            return Ok(Json(ApiResponse::error(
                "Invalid amount format".to_string(),
            )))
        }
    };

    // Get contract details to verify it exists and has burnable feature
    let ecosystem = &state.orobit_ecosystem;
    let contract = match ecosystem
        .get_contract_by_address(ContractAddress(contract_addr))
        .await
    {
        Some(c) => c,
        None => return Ok(Json(ApiResponse::error("Contract not found".to_string()))),
    };

    // Check if contract has burnable feature
    if !contract
        .metadata
        .features
        .get("burnable")
        .copied()
        .unwrap_or(false)
    {
        return Ok(Json(ApiResponse::error(
            "Contract does not support burning".to_string(),
        )));
    }

    // Burn tokens from the contract owner
    let owner = contract.deployer;
    let new_balance = {
        let mut token_balances = state.token_balances.write().await;
        let current_balance = token_balances
            .get(&(owner, contract_addr))
            .copied()
            .unwrap_or(0);

        if current_balance < amount {
            return Ok(Json(ApiResponse::error(format!(
                "Insufficient balance. Available: {}, Requested: {}",
                current_balance, amount
            ))));
        }

        let new_balance = current_balance - amount;
        token_balances.insert((owner, contract_addr), new_balance);

        tracing::info!(
            "🔥 Burned {} tokens for contract {} from owner {}. New balance: {}",
            amount,
            hex::encode(contract_addr),
            hex::encode(owner),
            new_balance
        );
        new_balance
    };

    // Persist token balance to storage
    if let Err(e) = state
        .storage_engine
        .save_token_balance(&owner, &contract_addr, new_balance)
        .await
    {
        tracing::warn!("Failed to persist token balance after burn: {}", e);
    }

    // Create transaction hash for the burn operation
    let tx_hash = format!(
        "burn-{}-{}",
        hex::encode(contract_addr),
        chrono::Utc::now().timestamp_millis()
    );

    Ok(Json(ApiResponse::success(TokenOperationResponse {
        success: true,
        transaction_hash: tx_hash,
        amount,
        message: format!("Successfully burned {} tokens", amount),
    })))
}

/// Airdrop tokens to multiple recipients
pub async fn airdrop_tokens(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AirdropRequest>,
) -> Result<Json<ApiResponse<TokenOperationResponse>>, StatusCode> {
    // Parse contract address
    let contract_addr = match parse_address(&request.contract_address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    // Parse amount per recipient
    let amount_per_recipient = match request.amount_per_recipient.parse::<u64>() {
        Ok(amt) if amt > 0 => amt,
        Ok(_) => {
            return Ok(Json(ApiResponse::error(
                "Amount must be greater than 0".to_string(),
            )))
        }
        Err(_) => {
            return Ok(Json(ApiResponse::error(
                "Invalid amount format".to_string(),
            )))
        }
    };

    // Validate recipients list
    if request.recipients.is_empty() {
        return Ok(Json(ApiResponse::error(
            "Recipients list cannot be empty".to_string(),
        )));
    }

    // Parse all recipient addresses
    let mut recipient_addrs = Vec::new();
    for recipient_str in &request.recipients {
        match parse_address(recipient_str) {
            Ok(addr) => recipient_addrs.push(addr),
            Err(e) => {
                return Ok(Json(ApiResponse::error(format!(
                    "Invalid recipient address '{}': {}",
                    recipient_str, e
                ))))
            }
        }
    }

    // Get contract details to verify it exists and has airdrop feature
    let ecosystem = &state.orobit_ecosystem;
    let contract = match ecosystem
        .get_contract_by_address(ContractAddress(contract_addr))
        .await
    {
        Some(c) => c,
        None => return Ok(Json(ApiResponse::error("Contract not found".to_string()))),
    };

    // Check if contract has airdrop feature
    if !contract
        .metadata
        .features
        .get("airdrop")
        .copied()
        .unwrap_or(false)
    {
        return Ok(Json(ApiResponse::error(
            "Contract does not support airdrops".to_string(),
        )));
    }

    // Calculate total amount needed
    let total_amount = amount_per_recipient.saturating_mul(recipient_addrs.len() as u64);

    // Check if owner has sufficient balance
    let owner = contract.deployer;
    let (new_owner_balance, recipient_balances) = {
        let mut token_balances = state.token_balances.write().await;
        let owner_balance = token_balances
            .get(&(owner, contract_addr))
            .copied()
            .unwrap_or(0);

        if owner_balance < total_amount {
            return Ok(Json(ApiResponse::error(format!(
                "Insufficient balance for airdrop. Required: {}, Available: {}",
                total_amount, owner_balance
            ))));
        }

        // Deduct from owner
        let new_owner_balance = owner_balance - total_amount;
        token_balances.insert((owner, contract_addr), new_owner_balance);

        // Distribute to recipients and collect new balances for persistence
        let mut recipient_balances = Vec::new();
        for recipient_addr in &recipient_addrs {
            let current_balance = token_balances
                .get(&(*recipient_addr, contract_addr))
                .copied()
                .unwrap_or(0);
            let new_balance = current_balance.saturating_add(amount_per_recipient);
            token_balances.insert((*recipient_addr, contract_addr), new_balance);
            recipient_balances.push((*recipient_addr, new_balance));

            tracing::debug!(
                "✈️ Airdropped {} tokens to {} for contract {}",
                amount_per_recipient,
                hex::encode(recipient_addr),
                hex::encode(contract_addr)
            );
        }

        tracing::info!(
            "✈️ Airdrop complete: {} tokens to {} recipients for contract {}. Total: {}",
            amount_per_recipient,
            recipient_addrs.len(),
            hex::encode(contract_addr),
            total_amount
        );

        (new_owner_balance, recipient_balances)
    };

    // Persist all balance changes to storage
    if let Err(e) = state
        .storage_engine
        .save_token_balance(&owner, &contract_addr, new_owner_balance)
        .await
    {
        tracing::warn!("Failed to persist owner balance after airdrop: {}", e);
    }
    for (recipient_addr, balance) in recipient_balances {
        if let Err(e) = state
            .storage_engine
            .save_token_balance(&recipient_addr, &contract_addr, balance)
            .await
        {
            tracing::warn!("Failed to persist recipient balance after airdrop: {}", e);
        }
    }

    // Create transaction hash for the airdrop operation
    let tx_hash = format!(
        "airdrop-{}-{}",
        hex::encode(contract_addr),
        chrono::Utc::now().timestamp_millis()
    );

    Ok(Json(ApiResponse::success(TokenOperationResponse {
        success: true,
        transaction_hash: tx_hash,
        amount: total_amount,
        message: format!(
            "Successfully airdropped {} tokens to {} recipients",
            amount_per_recipient,
            recipient_addrs.len()
        ),
    })))
}

/// Request body for pausing/resuming contract
#[derive(Debug, Deserialize)]
pub struct PauseRequest {
    pub contract_address: String,
    pub paused: bool,
}

/// Request body for updating reflection rate
#[derive(Debug, Deserialize)]
pub struct ReflectionRequest {
    pub contract_address: String,
    pub rate: String,
}

/// Pause or resume a contract
pub async fn pause_contract(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PauseRequest>,
) -> Result<Json<ApiResponse<TokenOperationResponse>>, StatusCode> {
    // Parse contract address
    let contract_addr = match parse_address(&request.contract_address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    // Get contract details to verify it exists and has pausable feature
    let ecosystem = &state.orobit_ecosystem;
    let contract = match ecosystem
        .get_contract_by_address(ContractAddress(contract_addr))
        .await
    {
        Some(c) => c,
        None => return Ok(Json(ApiResponse::error("Contract not found".to_string()))),
    };

    // Check if contract has pausable feature
    if !contract
        .metadata
        .features
        .get("pausable")
        .copied()
        .unwrap_or(false)
    {
        return Ok(Json(ApiResponse::error(
            "Contract does not support pausing".to_string(),
        )));
    }

    // In a real implementation, this would update the contract state
    // For now, we'll just log it
    tracing::info!(
        "⏸️ Contract {} pause state set to: {}",
        hex::encode(contract_addr),
        request.paused
    );

    // Create transaction hash for the pause operation
    let tx_hash = format!(
        "pause-{}-{}",
        hex::encode(contract_addr),
        chrono::Utc::now().timestamp_millis()
    );

    Ok(Json(ApiResponse::success(TokenOperationResponse {
        success: true,
        transaction_hash: tx_hash,
        amount: 0,
        message: format!(
            "Contract {} {}",
            if request.paused { "paused" } else { "resumed" },
            "successfully"
        ),
    })))
}

/// Update reflection rate for a contract
pub async fn update_reflection_rate(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ReflectionRequest>,
) -> Result<Json<ApiResponse<TokenOperationResponse>>, StatusCode> {
    // Parse contract address
    let contract_addr = match parse_address(&request.contract_address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    // Parse rate
    let rate = match request.rate.parse::<f64>() {
        Ok(r) if r >= 0.0 && r <= 10.0 => r,
        Ok(_) => {
            return Ok(Json(ApiResponse::error(
                "Rate must be between 0% and 10%".to_string(),
            )))
        }
        Err(_) => return Ok(Json(ApiResponse::error("Invalid rate format".to_string()))),
    };

    // Get contract details to verify it exists and has reflection feature
    let ecosystem = &state.orobit_ecosystem;
    let contract = match ecosystem
        .get_contract_by_address(ContractAddress(contract_addr))
        .await
    {
        Some(c) => c,
        None => return Ok(Json(ApiResponse::error("Contract not found".to_string()))),
    };

    // Check if contract has reflection feature
    if !contract
        .metadata
        .features
        .get("reflection")
        .copied()
        .unwrap_or(false)
    {
        return Ok(Json(ApiResponse::error(
            "Contract does not support reflection".to_string(),
        )));
    }

    // In a real implementation, this would update the contract configuration
    // For now, we'll just log it
    tracing::info!(
        "✨ Reflection rate for contract {} set to: {}%",
        hex::encode(contract_addr),
        rate
    );

    // Create transaction hash for the reflection update operation
    let tx_hash = format!(
        "reflection-{}-{}",
        hex::encode(contract_addr),
        chrono::Utc::now().timestamp_millis()
    );

    Ok(Json(ApiResponse::success(TokenOperationResponse {
        success: true,
        transaction_hash: tx_hash,
        amount: 0,
        message: format!("Reflection rate updated to {}%", rate),
    })))
}
