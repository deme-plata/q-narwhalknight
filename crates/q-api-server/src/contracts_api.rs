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
use crate::ContractEventRecord;
use crate::transaction_utils::{TransactionBuilder, submit_transaction};
use q_types::{Transaction, TxStatus, TokenAnnouncement};
use q_network::unified_network_manager::NetworkCommand;

/// v2.4.8: Token Social Profile - Decentralized social media links for custom tokens
/// Persisted to RocksDB and synced across nodes via gossipsub
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TokenSocialProfile {
    pub twitter: Option<String>,
    pub discord: Option<String>,
    pub telegram: Option<String>,
    pub website: Option<String>,
    pub github: Option<String>,
    pub medium: Option<String>,
    pub description: Option<String>,
    pub logo_url: Option<String>,
    pub updated_at: u64,
    pub owner_signature: Option<String>,
}
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

// ============ v2.4.2: TOKEN FEE CONFIGURATION ============
// Re-export types from q_storage to avoid duplication
// Implementations are in q_storage/lib.rs to satisfy Rust orphan rules
pub use q_storage::{TokenFeeConfig, TokenStakePosition, StakingTier};

/// Global storage for token fee configs (keyed by contract address hex)
pub type TokenFeeConfigStore = Arc<RwLock<HashMap<String, TokenFeeConfig>>>;

/// Global storage for staking positions (keyed by wallet+contract)
pub type TokenStakingStore = Arc<RwLock<HashMap<String, TokenStakePosition>>>;

/// Global storage for total reflected amounts per token
pub type TokenReflectionStore = Arc<RwLock<HashMap<String, u64>>>;

/// Global storage for total burned amounts per token
pub type TokenBurnStore = Arc<RwLock<HashMap<String, u64>>>;

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
/// v3.0.4: total_supply migrated to u128 for 24-decimal precision
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
    #[serde(serialize_with = "serialize_option_u128_as_string")]
    pub total_supply: Option<u128>, // v3.0.4: Migrated from u64 to u128
    pub decimals: Option<u32>,     // Add decimals for display
}

/// Helper to serialize Option<u128> as string for JSON (avoids JS 2^53 overflow)
fn serialize_option_u128_as_string<S>(value: &Option<u128>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match value {
        Some(v) => serializer.serialize_some(&v.to_string()),
        None => serializer.serialize_none(),
    }
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

/// v1.4.10: Response for contract events
#[derive(Debug, Serialize)]
pub struct ContractEventsResponse {
    pub contract_address: String,
    pub events: Vec<ContractEventRecord>,
    pub total_count: usize,
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
        // v1.4.10: Contract event history endpoint
        .route("/events/:address", get(get_contract_events))
        // User-specific endpoints
        .route("/user/:address/contracts", get(get_user_contracts))
        .route("/user/:address/deployments", get(get_user_deployments))
        // v2.4.2: Token staking endpoints
        .route("/:contract_address/stake", post(stake_tokens))
        .route("/:contract_address/unstake", post(unstake_tokens))
        .route("/:contract_address/claim-rewards", post(claim_staking_rewards))
        .route("/:contract_address/stake-info/:wallet_address", get(get_stake_info))
        .route("/:contract_address/pending-rewards/:wallet_address", get(get_pending_rewards))
        .route("/:contract_address/fee-config", get(get_fee_config))
        .route("/:contract_address/fee-config", post(update_fee_config))
        .route("/:contract_address/token-stats", get(get_token_stats))
        // v2.4.8: Social media profile endpoints
        .route("/:contract_address/social", get(get_social_profile))
        .route("/:contract_address/social", post(update_social_profile))
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

    // Ensure decimals parameter is set (default to 8 if not specified)
    let mut deployment_params = request.parameters.clone();
    if !deployment_params.contains_key("decimals") {
        deployment_params.insert("decimals".to_string(), serde_json::json!(8));
        tracing::info!("📊 Decimals not specified, defaulting to 8");
    }

    match ecosystem
        .deploy_contract(
            contract_type,
            deployer,
            deployment_params,
            deployment_options,
        )
        .await
    {
        Ok((request_id, contract_address)) => {
            // Deduct deployment cost from deployer's native QUG balance
            // v3.0.6-beta: Updated for 24 decimals (1 QUG = 10^24 base units)
            const DEPLOYMENT_COST: u128 = 1_000_000_000_000_000_000_000_000; // 1 QUG
            {
                let mut wallet_balances = state.wallet_balances.write().await;
                if let Some(balance) = wallet_balances.get_mut(&deployer) {
                    if *balance >= DEPLOYMENT_COST {
                        *balance -= DEPLOYMENT_COST;
                        tracing::info!(
                            "💸 Deducted {} QUG deployment cost from {}. New balance: {}",
                            DEPLOYMENT_COST as f64 / 1e24,
                            hex::encode(deployer),
                            *balance as f64 / 1e24
                        );

                        // ============================================================================
                        // 📡 v1.0.91-beta: PROPER CONTRACT DEPLOYMENT TRANSACTION HANDLING
                        // - Cryptographic transaction ID (SHA3-256)
                        // - Per-wallet nonce tracking (replay attack prevention)
                        // - Proper status: Pending -> InMempool -> Confirmed (not immediate)
                        // - Block production queue integration
                        // - Gossipsub broadcast with confirmation
                        // ============================================================================

                        // Get next nonce for this wallet (prevents replay attacks)
                        let nonce = state.nonce_tracker.get_and_increment(&deployer);

                        // Create transaction with proper cryptographic ID using transaction_utils
                        let transaction = TransactionBuilder::new()
                            .from(deployer)
                            .to(contract_address.0)
                            .amount(DEPLOYMENT_COST)
                            .fee(0) // Fee included in deployment cost
                            .data(format!("deploy:{}", request.contract_type).into_bytes())
                            .token_type(q_types::TokenType::QUG)
                            .fee_token_type(q_types::TokenType::QUGUSD)
                            .tx_type(q_types::TransactionType::ContractDeploy)
                            .build_with_nonce(nonce, chrono::Utc::now());

                        // Submit transaction properly:
                        // 1. Add to tx_pool with Pending status (not Confirmed!)
                        // 2. Add to production mempool for block inclusion
                        // 3. Broadcast to P2P network via gossipsub
                        let submission_result = submit_transaction(
                            transaction.clone(),
                            &state.tx_pool,
                            &state.tx_status,
                            state.production_mempool.as_ref(),
                            state.libp2p_discovery.as_ref(),
                        ).await;

                        tracing::info!(
                            "📝 [CONTRACT] {} deployment tx {} (nonce: {}, queued: {}, broadcast: {})",
                            request.contract_type,
                            &submission_result.tx_id_hex[..16],
                            nonce,
                            submission_result.queued_for_block,
                            submission_result.broadcast_success
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
                // Get decimals from parameters
                // Default to 8 decimals (same as QUG/QUGUSD) for consistency
                let decimals = request
                    .parameters
                    .get("decimals")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(8) as u32;

                // v1.0.49-beta: CRITICAL FIX - Convert human-readable to base units
                // User enters "1000000" (1 million tokens)
                // We store: 1000000 * 10^8 = 100,000,000,000,000 base units
                // This matches how liquidity and swaps work (8 decimal standard)
                // v2.7.9-beta: Changed from u64 to u128 for larger token supplies (up to 10^38)
                let decimal_multiplier = 10u128.pow(decimals);

                let initial_supply_result: Option<u128> = if let Some(supply_u64) =
                    initial_supply_val.as_u64()
                {
                    // Convert to base units: multiply by 10^decimals
                    let base_units = (supply_u64 as u128) * decimal_multiplier;
                    tracing::info!(
                        "✅ Token supply: {} display tokens × 10^{} = {} base units",
                        supply_u64,
                        decimals,
                        base_units
                    );
                    Some(base_units)
                } else if let Some(supply_str) = initial_supply_val.as_str() {
                    // v3.2.19-beta: String values are ALREADY in base units from frontend
                    // Frontend does: displayUnits * 10^decimals before sending
                    // So we should NOT multiply again here
                    match supply_str.parse::<u128>() {
                        Ok(base_units) => {
                            tracing::info!(
                                "✅ Token supply received: {} base units (frontend already converted)",
                                base_units
                            );
                            Some(base_units)
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
                        // Mint tokens to deployer's wallet (in base units)
                        let mut token_balances = state.token_balances.write().await;
                        token_balances.insert((deployer, contract_address.0), initial_supply);

                        // Calculate human-readable amount (for logging only)
                        let token_amount = initial_supply as f64 / decimal_multiplier as f64;
                        tracing::info!(
                            "💰 Minted {} base units ({} display tokens with {} decimals) to deployer {}",
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

            // ============================================================================
            // v2.3.7-beta: BROADCAST TOKEN DEPLOYMENT TO P2P NETWORK
            // Enables cross-node token discovery for true DEX decentralization
            // ============================================================================
            if let Some(ref libp2p_cmd_tx) = state.libp2p_command_tx {
                // Extract token metadata from deployment parameters
                let symbol = request.parameters.get("symbol")
                    .and_then(|v| v.as_str())
                    .unwrap_or("TOKEN")
                    .to_string();
                let name = request.parameters.get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&symbol)
                    .to_string();
                let decimals = request.parameters.get("decimals")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(8) as u8;
                let total_supply = request.parameters.get("initialSupply")
                    .or_else(|| request.parameters.get("initial_supply"))
                    .and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
                    .unwrap_or(0);

                // Create token announcement (without signature for now - signing requires Ed25519 key)
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);

                let announcement = TokenAnnouncement::new(
                    contract_address.0,
                    symbol.clone(),
                    name.clone(),
                    decimals,
                    total_supply * 10u64.pow(decimals as u32), // Convert to base units
                    deployer,
                    request.contract_type.clone(),
                    timestamp,
                );

                // Serialize and broadcast via P2P
                match postcard::to_allocvec(&announcement) {
                    Ok(announcement_bytes) => {
                        // Get network ID from environment (same pattern as main.rs)
                        let network_id = std::env::var("Q_NETWORK_ID")
                            .ok()
                            .and_then(|s| s.parse::<q_types::NetworkId>().ok())
                            .unwrap_or(q_types::NetworkId::TestnetPhase16);
                        let topic = network_id.contract_deployments_topic();
                        if let Err(e) = libp2p_cmd_tx.send(NetworkCommand::PublishTokenAnnouncement {
                            topic: topic.clone(),
                            announcement_bytes,
                        }) {
                            tracing::warn!("⚠️ [TOKEN P2P] Failed to send broadcast command: {}", e);
                        } else {
                            tracing::info!(
                                "🪙 [TOKEN P2P] Broadcast {} ({}) deployment to topic {}",
                                symbol, name, topic
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!("⚠️ [TOKEN P2P] Failed to serialize announcement: {}", e);
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
            // v1.4.9: Check both camelCase (initialSupply) and snake_case (initial_supply)
            // v3.0.4: Migrated to u128 for 24-decimal precision
            let total_supply = contract
                .deployment_params
                .get("initialSupply")
                .or_else(|| contract.deployment_params.get("initial_supply"))
                .and_then(|v| {
                    // Handle both number and string formats (u128 for large values)
                    v.as_u64().map(|n| n as u128)
                        .or_else(|| v.as_str().and_then(|s| s.parse::<u128>().ok()))
                });

            // v1.0.49-beta: FIXED - Default to 8 decimals (like Bitcoin satoshis)
            // This matches the standard throughout the system
            let decimals = contract
                .deployment_params
                .get("decimals")
                .and_then(|v| v.as_u64())
                .map(|d| d as u32)
                .or(Some(8)); // Default to 8 decimals (Bitcoin standard)

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

/// Get user contracts (identical function)
/// v1.0.49-beta: CRITICAL - This function is called from DexScreen.tsx for symbol mapping

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
            // v1.4.9: Check both camelCase (initialSupply) and snake_case (initial_supply)
            // v3.0.4: Migrated to u128 for 24-decimal precision
            let total_supply = contract
                .deployment_params
                .get("initialSupply")
                .or_else(|| contract.deployment_params.get("initial_supply"))
                .and_then(|v| {
                    // Handle both number and string formats (u128 for large values)
                    v.as_u64().map(|n| n as u128)
                        .or_else(|| v.as_str().and_then(|s| s.parse::<u128>().ok()))
                });

            // v1.0.49-beta: FIXED - Default to 8 decimals (like Bitcoin satoshis)
            let decimals = contract
                .deployment_params
                .get("decimals")
                .and_then(|v| v.as_u64())
                .map(|d| d as u32)
                .or(Some(8)); // Default to 8 decimals (Bitcoin standard)

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
    /// Balance as string to preserve precision for large numbers (JavaScript loses precision above 2^53)
    /// v2.7.9-beta: Changed to u128 for larger token supplies
    #[serde(serialize_with = "serialize_u128_as_string")]
    pub balance: u128,
}

/// Serialize u128 as string to preserve precision in JavaScript
/// v2.7.9-beta: Updated from u64 to u128
fn serialize_u128_as_string<S>(value: &u128, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&value.to_string())
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

    // v1.4.10: Log at INFO level to diagnose DEX balance mismatch
    tracing::info!(
        "🔍 [DEX BALANCE] Token balance query: wallet={}, token={}, balance={}",
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
        let new_balance = current_balance.saturating_add(amount as u128);
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

    // v1.4.10: Record mint event for event history
    let decimals = contract.deployment_params
        .get("decimals")
        .and_then(|v| v.as_u64())
        .unwrap_or(8) as u32;
    let display_amount = amount as f64 / 10f64.powi(decimals as i32);
    let event = ContractEventRecord {
        id: format!("mint-{}", chrono::Utc::now().timestamp_millis()),
        event_type: "mint".to_string(),
        amount: format!("{:.4}", display_amount),
        from: None,
        to: Some(hex::encode(owner)),
        recipients: None,
        timestamp: chrono::Utc::now().timestamp() as u64,
        tx_hash: tx_hash.clone(),
    };
    {
        let mut events = state.contract_events.write().await;
        let contract_key = hex::encode(contract_addr);
        events.entry(contract_key).or_insert_with(Vec::new).insert(0, event);
    }

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

        if current_balance < amount as u128 {
            return Ok(Json(ApiResponse::error(format!(
                "Insufficient balance. Available: {}, Requested: {}",
                current_balance, amount
            ))));
        }

        let new_balance = current_balance - amount as u128;
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

    // v1.4.10: Record burn event for event history
    let decimals = contract.deployment_params
        .get("decimals")
        .and_then(|v| v.as_u64())
        .unwrap_or(8) as u32;
    let display_amount = amount as f64 / 10f64.powi(decimals as i32);
    let event = ContractEventRecord {
        id: format!("burn-{}", chrono::Utc::now().timestamp_millis()),
        event_type: "burn".to_string(),
        amount: format!("{:.4}", display_amount),
        from: Some(hex::encode(owner)),
        to: None,
        recipients: None,
        timestamp: chrono::Utc::now().timestamp() as u64,
        tx_hash: tx_hash.clone(),
    };
    {
        let mut events = state.contract_events.write().await;
        let contract_key = hex::encode(contract_addr);
        events.entry(contract_key).or_insert_with(Vec::new).insert(0, event);
    }

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

        if owner_balance < total_amount as u128 {
            return Ok(Json(ApiResponse::error(format!(
                "Insufficient balance for airdrop. Required: {}, Available: {}",
                total_amount, owner_balance
            ))));
        }

        // Deduct from owner
        let new_owner_balance = owner_balance - total_amount as u128;
        token_balances.insert((owner, contract_addr), new_owner_balance);

        // Distribute to recipients and collect new balances for persistence
        let mut recipient_balances = Vec::new();
        for recipient_addr in &recipient_addrs {
            let current_balance = token_balances
                .get(&(*recipient_addr, contract_addr))
                .copied()
                .unwrap_or(0);
            let new_balance = current_balance.saturating_add(amount_per_recipient as u128);
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

    // v1.4.10: Record airdrop event for event history
    let decimals = contract.deployment_params
        .get("decimals")
        .and_then(|v| v.as_u64())
        .unwrap_or(8) as u32;
    let display_amount = amount_per_recipient as f64 / 10f64.powi(decimals as i32);
    let event = ContractEventRecord {
        id: format!("airdrop-{}", chrono::Utc::now().timestamp_millis()),
        event_type: "airdrop".to_string(),
        amount: format!("{:.4}", display_amount),
        from: Some(hex::encode(owner)),
        to: None,
        recipients: Some(recipient_addrs.len() as u32),
        timestamp: chrono::Utc::now().timestamp() as u64,
        tx_hash: tx_hash.clone(),
    };
    {
        let mut events = state.contract_events.write().await;
        let contract_key = hex::encode(contract_addr);
        events.entry(contract_key).or_insert_with(Vec::new).insert(0, event);
    }

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

/// v1.4.10: Get contract event history
pub async fn get_contract_events(
    State(state): State<Arc<AppState>>,
    Path(address): Path<String>,
) -> Result<Json<ApiResponse<ContractEventsResponse>>, StatusCode> {
    // Normalize the address (remove qnk prefix if present)
    let contract_key = if address.starts_with("qnk") {
        address[3..].to_string()
    } else {
        address.clone()
    };

    // Get events from storage
    let events = {
        let events_map = state.contract_events.read().await;
        events_map.get(&contract_key).cloned().unwrap_or_default()
    };

    let total_count = events.len();

    tracing::info!(
        "📜 Fetching events for contract {}: {} events found",
        contract_key,
        total_count
    );

    Ok(Json(ApiResponse::success(ContractEventsResponse {
        contract_address: address,
        events,
        total_count,
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

// ============ v2.4.2: TOKEN STAKING ENDPOINTS ============

/// Request to stake tokens
#[derive(Debug, Deserialize)]
pub struct StakeRequest {
    pub wallet_address: String,
    pub amount: String,
    pub lock_days: u64,
}

/// Response for staking operations
#[derive(Debug, Serialize)]
pub struct StakeResponse {
    pub success: bool,
    pub transaction_hash: String,
    pub stake_position: Option<StakePositionInfo>,
    pub message: String,
}

/// Stake position info for responses
#[derive(Debug, Serialize)]
pub struct StakePositionInfo {
    pub amount: f64,
    pub tier: String,
    pub apy: f64,
    pub start_time: u64,
    pub unlock_time: u64,
    pub pending_rewards: f64,
    pub total_rewards_claimed: f64,
    pub is_locked: bool,
    pub time_remaining_seconds: u64,
}

/// Token statistics response
#[derive(Debug, Serialize)]
pub struct TokenStatsResponse {
    pub contract_address: String,
    pub symbol: String,
    pub total_supply: f64,
    pub circulating_supply: f64,
    pub total_staked: f64,
    pub total_burned: f64,
    pub total_reflected: f64,
    pub holder_count: u64,
    pub staker_count: u64,
    pub fee_config: TokenFeeConfig,
}

/// Request to update fee config (owner only)
#[derive(Debug, Deserialize)]
pub struct UpdateFeeConfigRequest {
    pub wallet_address: String,
    pub enabled: Option<bool>,
    pub reflection_fee_bps: Option<u64>,
    pub burn_fee_bps: Option<u64>,
    pub liquidity_fee_bps: Option<u64>,
    pub dev_fee_bps: Option<u64>,
}

/// Stake tokens in a custom token contract
pub async fn stake_tokens(
    State(state): State<Arc<AppState>>,
    Path(contract_address): Path<String>,
    Json(request): Json<StakeRequest>,
) -> Result<Json<ApiResponse<StakeResponse>>, StatusCode> {
    tracing::info!("🔒 [STAKING] Stake request for contract {}: {} tokens for {} days",
        contract_address, request.amount, request.lock_days);

    // Parse contract address
    let contract_addr = match parse_address(&contract_address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    // Parse wallet address
    let wallet_addr = match parse_address(&request.wallet_address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    // Parse amount
    let amount_f64: f64 = match request.amount.parse() {
        Ok(a) => a,
        Err(_) => return Ok(Json(ApiResponse::error("Invalid amount".to_string()))),
    };
    let amount = (amount_f64 * 100_000_000.0) as u64;

    if amount == 0 {
        return Ok(Json(ApiResponse::error("Amount must be greater than 0".to_string())));
    }

    // Check token balance
    let token_balances = state.token_balances.read().await;
    let balance_key = (wallet_addr, contract_addr);
    let current_balance = token_balances.get(&balance_key).copied().unwrap_or(0);
    drop(token_balances);

    if current_balance < amount as u128 {
        return Ok(Json(ApiResponse::error(format!(
            "Insufficient balance. Have: {}, Need: {}",
            current_balance as f64 / 1e24,
            amount_f64
        ))));
    }

    // Calculate tier
    let tier = StakingTier::from_days(request.lock_days);
    let current_time = current_timestamp();
    let unlock_time = current_time + tier.lock_period_seconds();

    // Create stake position
    let stake_key = format!("{}:{}", request.wallet_address.to_lowercase(), contract_address.to_lowercase());

    let mut staking_store = state.token_staking_positions.write().await;

    // Check if already staking
    if let Some(existing) = staking_store.get(&stake_key) {
        if current_time < existing.unlock_time {
            return Ok(Json(ApiResponse::error(
                "Already have an active stake. Unstake first or wait for unlock.".to_string()
            )));
        }
    }

    // Lock tokens (deduct from balance)
    let mut token_balances = state.token_balances.write().await;
    let new_balance = current_balance - amount as u128;
    token_balances.insert(balance_key, new_balance);
    drop(token_balances);

    // Persist balance change
    if let Err(e) = state.storage_engine.save_token_balance(&wallet_addr, &contract_addr, new_balance).await {
        tracing::warn!("Failed to persist stake balance change: {}", e);
    }

    // Create stake position
    let stake_position = TokenStakePosition {
        wallet_address: request.wallet_address.clone(),
        contract_address: contract_address.clone(),
        amount,
        tier,
        start_time: current_time,
        unlock_time,
        last_reward_claim: current_time,
        total_rewards_claimed: 0,
    };

    staking_store.insert(stake_key.clone(), stake_position.clone());
    drop(staking_store);

    // Persist stake position
    if let Err(e) = state.storage_engine.save_stake_position(&stake_key, &stake_position).await {
        tracing::warn!("Failed to persist stake position: {}", e);
    }

    let tx_hash = format!("stake-{}-{}", hex::encode(contract_addr), current_time);

    tracing::info!("✅ [STAKING] {} staked {} tokens in {} tier (unlocks at {})",
        request.wallet_address, amount_f64, tier.name(), unlock_time);

    Ok(Json(ApiResponse::success(StakeResponse {
        success: true,
        transaction_hash: tx_hash,
        stake_position: Some(StakePositionInfo {
            amount: amount_f64,
            tier: tier.name().to_string(),
            apy: tier.apy_bps() as f64 / 100.0,
            start_time: current_time,
            unlock_time,
            pending_rewards: 0.0,
            total_rewards_claimed: 0.0,
            is_locked: true,
            time_remaining_seconds: unlock_time - current_time,
        }),
        message: format!("Successfully staked {} tokens in {} tier", amount_f64, tier.name()),
    })))
}

/// Unstake tokens from a custom token contract
pub async fn unstake_tokens(
    State(state): State<Arc<AppState>>,
    Path(contract_address): Path<String>,
    Json(request): Json<StakeRequest>,
) -> Result<Json<ApiResponse<StakeResponse>>, StatusCode> {
    tracing::info!("🔓 [STAKING] Unstake request for contract {}", contract_address);

    // Parse addresses
    let contract_addr = match parse_address(&contract_address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    let wallet_addr = match parse_address(&request.wallet_address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    let stake_key = format!("{}:{}", request.wallet_address.to_lowercase(), contract_address.to_lowercase());
    let current_time = current_timestamp();

    let mut staking_store = state.token_staking_positions.write().await;

    let stake = match staking_store.get(&stake_key) {
        Some(s) => s.clone(),
        None => return Ok(Json(ApiResponse::error("No active stake found".to_string()))),
    };

    // Check if still locked
    if current_time < stake.unlock_time {
        let remaining = stake.unlock_time - current_time;
        return Ok(Json(ApiResponse::error(format!(
            "Stake still locked. {} seconds remaining",
            remaining
        ))));
    }

    // Calculate pending rewards
    let pending_rewards = calculate_pending_rewards_internal(&stake);

    // Return staked amount + rewards to balance
    let mut token_balances = state.token_balances.write().await;
    let balance_key = (wallet_addr, contract_addr);
    let current_balance = token_balances.get(&balance_key).copied().unwrap_or(0);
    let new_balance = current_balance + stake.amount as u128 + pending_rewards as u128;
    token_balances.insert(balance_key, new_balance);
    drop(token_balances);

    // Persist balance change
    if let Err(e) = state.storage_engine.save_token_balance(&wallet_addr, &contract_addr, new_balance).await {
        tracing::warn!("Failed to persist unstake balance change: {}", e);
    }

    // Remove stake position
    staking_store.remove(&stake_key);
    drop(staking_store);

    // Remove from persistent storage
    if let Err(e) = state.storage_engine.delete_stake_position(&stake_key).await {
        tracing::warn!("Failed to delete stake position from storage: {}", e);
    }

    let tx_hash = format!("unstake-{}-{}", hex::encode(contract_addr), current_time);
    let total_returned = (stake.amount + pending_rewards) as f64 / 1e24;

    tracing::info!("✅ [STAKING] {} unstaked {} tokens (+ {} rewards)",
        request.wallet_address,
        stake.amount as f64 / 1e24,
        pending_rewards as f64 / 1e24);

    Ok(Json(ApiResponse::success(StakeResponse {
        success: true,
        transaction_hash: tx_hash,
        stake_position: None,
        message: format!("Successfully unstaked {} tokens (including {} in rewards)",
            total_returned, pending_rewards as f64 / 1e24),
    })))
}

/// Claim staking rewards without unstaking
pub async fn claim_staking_rewards(
    State(state): State<Arc<AppState>>,
    Path(contract_address): Path<String>,
    Json(request): Json<StakeRequest>,
) -> Result<Json<ApiResponse<StakeResponse>>, StatusCode> {
    tracing::info!("💰 [STAKING] Claim rewards request for contract {}", contract_address);

    // Parse addresses
    let contract_addr = match parse_address(&contract_address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    let wallet_addr = match parse_address(&request.wallet_address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    let stake_key = format!("{}:{}", request.wallet_address.to_lowercase(), contract_address.to_lowercase());
    let current_time = current_timestamp();

    let mut staking_store = state.token_staking_positions.write().await;

    let stake = match staking_store.get_mut(&stake_key) {
        Some(s) => s,
        None => return Ok(Json(ApiResponse::error("No active stake found".to_string()))),
    };

    // Calculate pending rewards
    let pending_rewards = calculate_pending_rewards_internal(stake);

    if pending_rewards == 0 {
        return Ok(Json(ApiResponse::error("No rewards to claim".to_string())));
    }

    // Update stake position
    stake.last_reward_claim = current_time;
    stake.total_rewards_claimed += pending_rewards;
    let updated_stake = stake.clone();
    drop(staking_store);

    // Add rewards to balance
    let mut token_balances = state.token_balances.write().await;
    let balance_key = (wallet_addr, contract_addr);
    let current_balance = token_balances.get(&balance_key).copied().unwrap_or(0);
    let new_balance = current_balance + pending_rewards as u128;
    token_balances.insert(balance_key, new_balance);
    drop(token_balances);

    // Persist changes
    if let Err(e) = state.storage_engine.save_token_balance(&wallet_addr, &contract_addr, new_balance).await {
        tracing::warn!("Failed to persist reward claim balance: {}", e);
    }

    let stake_key_for_save = format!("{}:{}", request.wallet_address.to_lowercase(), contract_address.to_lowercase());
    if let Err(e) = state.storage_engine.save_stake_position(&stake_key_for_save, &updated_stake).await {
        tracing::warn!("Failed to persist stake position update: {}", e);
    }

    let tx_hash = format!("claim-{}-{}", hex::encode(contract_addr), current_time);
    let rewards_f64 = pending_rewards as f64 / 1e24;

    tracing::info!("✅ [STAKING] {} claimed {} in rewards", request.wallet_address, rewards_f64);

    Ok(Json(ApiResponse::success(StakeResponse {
        success: true,
        transaction_hash: tx_hash,
        stake_position: Some(StakePositionInfo {
            amount: updated_stake.amount as f64 / 1e24,
            tier: updated_stake.tier.name().to_string(),
            apy: updated_stake.tier.apy_bps() as f64 / 100.0,
            start_time: updated_stake.start_time,
            unlock_time: updated_stake.unlock_time,
            pending_rewards: 0.0,
            total_rewards_claimed: updated_stake.total_rewards_claimed as f64 / 1e24,
            is_locked: current_time < updated_stake.unlock_time,
            time_remaining_seconds: updated_stake.unlock_time.saturating_sub(current_time),
        }),
        message: format!("Successfully claimed {} in rewards", rewards_f64),
    })))
}

/// Get stake info for a wallet
pub async fn get_stake_info(
    State(state): State<Arc<AppState>>,
    Path((contract_address, wallet_address)): Path<(String, String)>,
) -> Result<Json<ApiResponse<StakePositionInfo>>, StatusCode> {
    let stake_key = format!("{}:{}", wallet_address.to_lowercase(), contract_address.to_lowercase());
    let current_time = current_timestamp();

    let staking_store = state.token_staking_positions.read().await;

    match staking_store.get(&stake_key) {
        Some(stake) => {
            let pending_rewards = calculate_pending_rewards_internal(stake);
            Ok(Json(ApiResponse::success(StakePositionInfo {
                amount: stake.amount as f64 / 1e24,
                tier: stake.tier.name().to_string(),
                apy: stake.tier.apy_bps() as f64 / 100.0,
                start_time: stake.start_time,
                unlock_time: stake.unlock_time,
                pending_rewards: pending_rewards as f64 / 1e24,
                total_rewards_claimed: stake.total_rewards_claimed as f64 / 1e24,
                is_locked: current_time < stake.unlock_time,
                time_remaining_seconds: stake.unlock_time.saturating_sub(current_time),
            })))
        }
        None => Ok(Json(ApiResponse::error("No active stake found".to_string()))),
    }
}

/// Get pending rewards for a wallet
pub async fn get_pending_rewards(
    State(state): State<Arc<AppState>>,
    Path((contract_address, wallet_address)): Path<(String, String)>,
) -> Result<Json<ApiResponse<f64>>, StatusCode> {
    let stake_key = format!("{}:{}", wallet_address.to_lowercase(), contract_address.to_lowercase());

    let staking_store = state.token_staking_positions.read().await;

    match staking_store.get(&stake_key) {
        Some(stake) => {
            let pending_rewards = calculate_pending_rewards_internal(stake);
            Ok(Json(ApiResponse::success(pending_rewards as f64 / 1e24)))
        }
        None => Ok(Json(ApiResponse::success(0.0))),
    }
}

/// Get fee configuration for a token
pub async fn get_fee_config(
    State(state): State<Arc<AppState>>,
    Path(contract_address): Path<String>,
) -> Result<Json<ApiResponse<TokenFeeConfig>>, StatusCode> {
    let fee_configs = state.token_fee_configs.read().await;

    match fee_configs.get(&contract_address.to_lowercase()) {
        Some(config) => Ok(Json(ApiResponse::success(config.clone()))),
        None => Ok(Json(ApiResponse::success(TokenFeeConfig::default()))),
    }
}

/// Update fee configuration (owner only)
pub async fn update_fee_config(
    State(state): State<Arc<AppState>>,
    Path(contract_address): Path<String>,
    Json(request): Json<UpdateFeeConfigRequest>,
) -> Result<Json<ApiResponse<TokenFeeConfig>>, StatusCode> {
    tracing::info!("⚙️ [FEES] Update fee config request for {}", contract_address);

    // Parse contract address
    let contract_addr = match parse_address(&contract_address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    // Parse wallet address
    let wallet_addr = match parse_address(&request.wallet_address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    // Check if caller is contract owner
    let ecosystem = &state.orobit_ecosystem;
    let contract = match ecosystem
        .get_contract_by_address(ContractAddress(contract_addr))
        .await
    {
        Some(c) => c,
        None => return Ok(Json(ApiResponse::error("Contract not found".to_string()))),
    };

    if contract.deployer != wallet_addr {
        return Ok(Json(ApiResponse::error("Only contract owner can update fee config".to_string())));
    }

    // Validate total fee doesn't exceed 10%
    let total_bps = request.reflection_fee_bps.unwrap_or(0)
        + request.burn_fee_bps.unwrap_or(0)
        + request.liquidity_fee_bps.unwrap_or(0)
        + request.dev_fee_bps.unwrap_or(0);

    if total_bps > 1000 {
        return Ok(Json(ApiResponse::error("Total fees cannot exceed 10% (1000 basis points)".to_string())));
    }

    let mut fee_configs = state.token_fee_configs.write().await;

    let config = fee_configs
        .entry(contract_address.to_lowercase())
        .or_insert_with(TokenFeeConfig::default);

    // Update only provided fields
    if let Some(enabled) = request.enabled {
        config.enabled = enabled;
    }
    if let Some(reflection) = request.reflection_fee_bps {
        config.reflection_fee_bps = reflection;
    }
    if let Some(burn) = request.burn_fee_bps {
        config.burn_fee_bps = burn;
    }
    if let Some(liquidity) = request.liquidity_fee_bps {
        config.liquidity_fee_bps = liquidity;
    }
    if let Some(dev) = request.dev_fee_bps {
        config.dev_fee_bps = dev;
    }

    let updated_config = config.clone();
    drop(fee_configs);

    // Persist fee config
    if let Err(e) = state.storage_engine.save_fee_config(&contract_address.to_lowercase(), &updated_config).await {
        tracing::warn!("Failed to persist fee config: {}", e);
    }

    tracing::info!("✅ [FEES] Fee config updated for {}: {:?}", contract_address, updated_config);

    Ok(Json(ApiResponse::success(updated_config)))
}

/// Get token statistics
pub async fn get_token_stats(
    State(state): State<Arc<AppState>>,
    Path(contract_address): Path<String>,
) -> Result<Json<ApiResponse<TokenStatsResponse>>, StatusCode> {
    // Parse contract address
    let contract_addr = match parse_address(&contract_address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    // Get contract info
    let ecosystem = &state.orobit_ecosystem;
    let contract = match ecosystem
        .get_contract_by_address(ContractAddress(contract_addr))
        .await
    {
        Some(c) => c,
        None => return Ok(Json(ApiResponse::error("Contract not found".to_string()))),
    };

    let symbol = contract.metadata.symbol.clone().unwrap_or_else(|| "TOKEN".to_string());

    // Get fee config
    let fee_configs = state.token_fee_configs.read().await;
    let fee_config = fee_configs.get(&contract_address.to_lowercase())
        .cloned()
        .unwrap_or_default();
    drop(fee_configs);

    // Count holders and calculate totals
    // v2.7.9-beta: Changed total_supply to u128 for larger token supplies
    let token_balances = state.token_balances.read().await;
    let mut total_supply: u128 = 0;
    let mut holder_count: u64 = 0;

    for ((_, token_addr), balance) in token_balances.iter() {
        if *token_addr == contract_addr && *balance > 0 {
            total_supply += *balance;
            holder_count += 1;
        }
    }
    drop(token_balances);

    // Count stakers and total staked
    let staking_store = state.token_staking_positions.read().await;
    let mut total_staked: u64 = 0;
    let mut staker_count: u64 = 0;

    for (key, stake) in staking_store.iter() {
        if key.ends_with(&format!(":{}", contract_address.to_lowercase())) {
            total_staked += stake.amount;
            staker_count += 1;
        }
    }
    drop(staking_store);

    // Get burn/reflection totals
    let burn_store = state.token_burn_totals.read().await;
    let total_burned = burn_store.get(&contract_address.to_lowercase()).copied().unwrap_or(0);
    drop(burn_store);

    let reflection_store = state.token_reflection_totals.read().await;
    let total_reflected = reflection_store.get(&contract_address.to_lowercase()).copied().unwrap_or(0);
    drop(reflection_store);

    Ok(Json(ApiResponse::success(TokenStatsResponse {
        contract_address,
        symbol,
        total_supply: (total_supply + total_staked as u128) as f64 / 1e24,
        circulating_supply: total_supply as f64 / 1e24,
        total_staked: total_staked as f64 / 1e24,
        total_burned: total_burned as f64 / 1e24,
        total_reflected: total_reflected as f64 / 1e24,
        holder_count,
        staker_count,
        fee_config,
    })))
}

/// Calculate pending rewards for a stake position
fn calculate_pending_rewards_internal(stake: &TokenStakePosition) -> u64 {
    let current_time = current_timestamp();
    let time_staked = current_time.saturating_sub(stake.last_reward_claim);
    let seconds_per_year: u64 = 365 * 24 * 3600;

    // Calculate rewards based on APY and time
    let annual_reward = (stake.amount * stake.tier.apy_bps()) / 10000;
    let pending = (annual_reward * time_staked) / seconds_per_year;

    pending
}

// ============ v2.4.8: SOCIAL MEDIA PROFILE ENDPOINTS ============

/// Request body for updating social profile
#[derive(Debug, Deserialize)]
pub struct UpdateSocialProfileRequest {
    pub twitter: Option<String>,
    pub discord: Option<String>,
    pub telegram: Option<String>,
    pub website: Option<String>,
    pub github: Option<String>,
    pub medium: Option<String>,
    pub description: Option<String>,
    pub logo_url: Option<String>,
    /// Wallet address of the owner (for verification)
    pub owner_address: String,
    /// Signature proving ownership
    pub signature: Option<String>,
}

/// Response for social profile
#[derive(Debug, Serialize)]
pub struct SocialProfileResponse {
    pub contract_address: String,
    pub twitter: Option<String>,
    pub discord: Option<String>,
    pub telegram: Option<String>,
    pub website: Option<String>,
    pub github: Option<String>,
    pub medium: Option<String>,
    pub description: Option<String>,
    pub logo_url: Option<String>,
    pub updated_at: u64,
}

/// Get social media profile for a token contract
pub async fn get_social_profile(
    State(state): State<Arc<AppState>>,
    Path(contract_address): Path<String>,
) -> Result<Json<ApiResponse<SocialProfileResponse>>, StatusCode> {
    let key = contract_address.to_lowercase();

    // Check in-memory cache first
    let profiles = state.token_social_profiles.read().await;
    if let Some(profile) = profiles.get(&key) {
        return Ok(Json(ApiResponse::success(SocialProfileResponse {
            contract_address: contract_address.clone(),
            twitter: profile.twitter.clone(),
            discord: profile.discord.clone(),
            telegram: profile.telegram.clone(),
            website: profile.website.clone(),
            github: profile.github.clone(),
            medium: profile.medium.clone(),
            description: profile.description.clone(),
            logo_url: profile.logo_url.clone(),
            updated_at: profile.updated_at,
        })));
    }
    drop(profiles);

    // Try loading from RocksDB (no in-memory cache to avoid type conflicts)
    match state.storage_engine.load_social_profile(&key).await {
        Ok(Some(data)) => {
            if let Ok(profile) = serde_json::from_slice::<TokenSocialProfile>(&data) {
                return Ok(Json(ApiResponse::success(SocialProfileResponse {
                    contract_address,
                    twitter: profile.twitter,
                    discord: profile.discord,
                    telegram: profile.telegram,
                    website: profile.website,
                    github: profile.github,
                    medium: profile.medium,
                    description: profile.description,
                    logo_url: profile.logo_url,
                    updated_at: profile.updated_at,
                })));
            }
        }
        Ok(None) => {}
        Err(e) => {
            tracing::warn!("Failed to load social profile from storage: {}", e);
        }
    }

    // Return empty profile if not found
    Ok(Json(ApiResponse::success(SocialProfileResponse {
        contract_address,
        twitter: None,
        discord: None,
        telegram: None,
        website: None,
        github: None,
        medium: None,
        description: None,
        logo_url: None,
        updated_at: 0,
    })))
}

/// Update social media profile for a token contract
pub async fn update_social_profile(
    State(state): State<Arc<AppState>>,
    Path(contract_address): Path<String>,
    Json(request): Json<UpdateSocialProfileRequest>,
) -> Result<Json<ApiResponse<SocialProfileResponse>>, StatusCode> {
    let key = contract_address.to_lowercase();

    // TODO: Verify ownership via signature
    // For now, we trust the caller (frontend has already validated session)

    let profile = TokenSocialProfile {
        twitter: request.twitter.clone(),
        discord: request.discord.clone(),
        telegram: request.telegram.clone(),
        website: request.website.clone(),
        github: request.github.clone(),
        medium: request.medium.clone(),
        description: request.description.clone(),
        logo_url: request.logo_url.clone(),
        updated_at: current_timestamp(),
        owner_signature: request.signature.clone(),
    };

    // Save to RocksDB for persistence (no in-memory cache to avoid type conflicts)
    if let Ok(data) = serde_json::to_vec(&profile) {
        if let Err(e) = state.storage_engine.save_social_profile(&key, &data).await {
            tracing::error!("Failed to save social profile to storage: {}", e);
            return Ok(Json(ApiResponse::error(format!("Storage error: {}", e))));
        }
    }

    // Broadcast via gossipsub to sync across nodes
    if let Some(tx) = &state.libp2p_command_tx {
        let network_id = std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| "testnet-phase16".to_string());
        let topic = format!("/qnk/{}/token-social", network_id);

        let message = serde_json::json!({
            "type": "token_social_update",
            "contract_address": key.clone(),
            "profile": profile,
        });

        if let Ok(profile_bytes) = serde_json::to_vec(&message) {
            let _ = tx.send(q_network::NetworkCommand::PublishTokenSocial {
                topic,
                contract_address: key.clone(),
                profile_bytes,
            });
            tracing::info!("📡 Broadcast social profile update for {} via P2P", key);
        }
    }

    tracing::info!("📱 Updated social profile for token {}", key);

    Ok(Json(ApiResponse::success(SocialProfileResponse {
        contract_address,
        twitter: request.twitter,
        discord: request.discord,
        telegram: request.telegram,
        website: request.website,
        github: request.github,
        medium: request.medium,
        description: request.description,
        logo_url: request.logo_url,
        updated_at: profile.updated_at,
    })))
}
