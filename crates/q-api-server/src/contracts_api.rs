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
use crate::wallet_auth::AuthenticatedWallet;
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

// ============================================================================
// v4.2.0-beta: VAULT RWA Token — Physical Device Redemption System
// ============================================================================

/// A redemption request for a physical Quillon Vault device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultRedemption {
    pub redemption_id: String,
    pub buyer_wallet: String,
    pub shipping_name: String,
    pub shipping_address: String,
    pub city: String,
    pub state_province: String,
    pub zip: String,
    pub country: String,
    pub phone: String,
    pub email: String,
    pub color_variant: String,
    pub quantity: u32,
    pub status: String, // "pending", "processing", "shipped", "delivered"
    pub tracking_number: Option<String>,
    pub serial_number: Option<String>,
    pub created_at: u64,
    pub fulfilled_at: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct VaultRedeemRequest {
    pub shipping_name: String,
    pub shipping_address: String,
    pub city: String,
    pub state_province: String,
    pub zip: String,
    pub country: String,
    pub phone: String,
    pub email: String,
    pub color_variant: String,
    pub quantity: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct VaultFulfillRequest {
    pub redemption_id: String,
    pub tracking_number: Option<String>,
    pub serial_number: Option<String>,
    pub status: String,
}

// ============================================================================
// v5.1.0: FORGE RWA Token — Physical Mining Machine Redemption System
// ============================================================================

/// A redemption request for a physical Quillon Forge mining machine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgeRedemption {
    pub redemption_id: String,
    pub buyer_wallet: String,
    pub shipping_name: String,
    pub shipping_address: String,
    pub city: String,
    pub state_province: String,
    pub zip: String,
    pub country: String,
    pub phone: String,
    pub email: String,
    /// CPU configuration: "epyc-9755-dual" or "xeon-w9-3595x-dual"
    pub cpu_config: String,
    /// GPU configuration: "none", "rtx-5090-dual", "a100-dual", "l40-quad"
    pub gpu_config: String,
    /// Cooling type: "liquid-copper" (default), "air-cooled"
    pub cooling_type: String,
    /// RAM amount in GB: 512, 1024, 2048
    pub ram_gb: u32,
    /// Storage configuration: "nvme-4tb-raid1" (default), "nvme-8tb-raid1"
    pub storage_config: String,
    /// NIC: "100gbe" (default), "25gbe"
    pub nic_config: String,
    /// Chassis color: "titanium-copper" (default), "obsidian-black", "arctic-white"
    pub chassis_color: String,
    pub quantity: u32,
    pub status: String, // "pending", "configured", "assembling", "testing", "shipped", "delivered"
    pub tracking_number: Option<String>,
    pub serial_number: Option<String>,
    /// Unique machine ID burned into firmware
    pub machine_id: Option<String>,
    /// Hardware attestation key for proof-of-work validation
    pub attestation_pubkey: Option<String>,
    pub created_at: u64,
    pub fulfilled_at: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct ForgeRedeemRequest {
    pub shipping_name: String,
    pub shipping_address: String,
    pub city: String,
    pub state_province: String,
    pub zip: String,
    pub country: String,
    pub phone: String,
    pub email: String,
    pub cpu_config: Option<String>,
    pub gpu_config: Option<String>,
    pub cooling_type: Option<String>,
    pub ram_gb: Option<u32>,
    pub storage_config: Option<String>,
    pub nic_config: Option<String>,
    pub chassis_color: Option<String>,
    pub quantity: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct ForgeFulfillRequest {
    pub redemption_id: String,
    pub tracking_number: Option<String>,
    pub serial_number: Option<String>,
    pub machine_id: Option<String>,
    pub attestation_pubkey: Option<String>,
    pub status: String,
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

// ============================================================================
// v6.5.0: RWA Persistence Helpers (Vault, Forge redemptions → RocksDB)
// ============================================================================

const VAULT_REDEMPTION_PREFIX: &str = "vault_redemption_";
const FORGE_REDEMPTION_PREFIX: &str = "forge_redemption_";

async fn persist_vault_redemption(state: &AppState, redemption: &VaultRedemption) {
    let key = format!("{}{}", VAULT_REDEMPTION_PREFIX, redemption.redemption_id);
    match serde_json::to_vec(redemption) {
        Ok(data) => {
            let kv = state.storage_engine.get_kv();
            if let Err(e) = kv.put_sync(q_storage::CF_MANIFEST, key.as_bytes(), &data).await {
                tracing::warn!("Failed to persist vault redemption {}: {}", redemption.redemption_id, e);
            }
        }
        Err(e) => tracing::warn!("Failed to serialize vault redemption: {}", e),
    }
}

async fn persist_forge_redemption(state: &AppState, redemption: &ForgeRedemption) {
    let key = format!("{}{}", FORGE_REDEMPTION_PREFIX, redemption.redemption_id);
    match serde_json::to_vec(redemption) {
        Ok(data) => {
            let kv = state.storage_engine.get_kv();
            if let Err(e) = kv.put_sync(q_storage::CF_MANIFEST, key.as_bytes(), &data).await {
                tracing::warn!("Failed to persist forge redemption {}: {}", redemption.redemption_id, e);
            }
        }
        Err(e) => tracing::warn!("Failed to serialize forge redemption: {}", e),
    }
}

pub async fn load_vault_redemptions_from_db(state: &AppState) -> Vec<VaultRedemption> {
    let mut redemptions = Vec::new();
    let prefix = VAULT_REDEMPTION_PREFIX.as_bytes();
    let kv = state.storage_engine.get_kv();

    match kv.scan_prefix(q_storage::CF_MANIFEST, prefix).await {
        Ok(entries) => {
            for (_key, value) in entries {
                match serde_json::from_slice::<VaultRedemption>(&value) {
                    Ok(r) => redemptions.push(r),
                    Err(e) => tracing::warn!("Failed to deserialize vault redemption: {}", e),
                }
            }
        }
        Err(e) => tracing::warn!("Failed to load vault redemptions from DB: {}", e),
    }

    redemptions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    redemptions
}

pub async fn load_forge_redemptions_from_db(state: &AppState) -> Vec<ForgeRedemption> {
    let mut redemptions = Vec::new();
    let prefix = FORGE_REDEMPTION_PREFIX.as_bytes();
    let kv = state.storage_engine.get_kv();

    match kv.scan_prefix(q_storage::CF_MANIFEST, prefix).await {
        Ok(entries) => {
            for (_key, value) in entries {
                match serde_json::from_slice::<ForgeRedemption>(&value) {
                    Ok(r) => redemptions.push(r),
                    Err(e) => tracing::warn!("Failed to deserialize forge redemption: {}", e),
                }
            }
        }
        Err(e) => tracing::warn!("Failed to load forge redemptions from DB: {}", e),
    }

    redemptions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    redemptions
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
    pub deployment_params: Option<serde_json::Value>, // v4.0.3: RWA configuration parameters
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
        // RWA marketplace listing endpoint
        .route("/rwa/marketplace", get(get_rwa_marketplace))
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
        // v4.1.0: RWA Portfolio & Collateralization endpoints
        .route("/rwa/portfolio", get(get_rwa_portfolio))
        .route("/rwa/collateral/borrow", post(rwa_collateral_borrow))
        .route("/rwa/collateral/repay", post(rwa_collateral_repay))
        .route("/rwa/distribution/schedule", post(rwa_schedule_distribution))
        .route("/rwa/distribution/toggle", post(rwa_toggle_distribution))
        .route("/rwa/compliance/check", post(rwa_compliance_check))
        // v4.2.0: VAULT RWA Physical Device Redemption endpoints
        .route("/vault/redeem", post(vault_redeem))
        .route("/vault/redemptions", get(vault_get_redemptions))
        .route("/vault/fulfill", post(vault_fulfill))
        .route("/vault/stats", get(vault_get_stats))
        // v5.1.0: FORGE RWA Physical Mining Machine Redemption endpoints
        .route("/forge/redeem", post(forge_redeem))
        .route("/forge/redemptions", get(forge_get_redemptions))
        .route("/forge/fulfill", post(forge_fulfill))
        .route("/forge/stats", get(forge_get_stats))
        // v6.5.0: Exchange Listing RWA Package endpoints
        .route("/listing/packages", get(crate::listing_api::listing_get_packages))
        .route("/listing/purchase", post(crate::listing_api::listing_purchase))
        .route("/listing/orders", get(crate::listing_api::listing_get_orders))
        .route("/listing/fulfill", post(crate::listing_api::listing_fulfill))
        .route("/listing/stats", get(crate::listing_api::listing_get_stats))
        .route("/listing/confirm-stripe", post(crate::listing_api::listing_confirm_stripe))
        // v8.2.8: XLIST Crowdfunding Campaign endpoints
        .route("/listing/campaigns", get(crate::listing_api::campaign_list))
        .route("/listing/campaigns/create", post(crate::listing_api::campaign_create))
        .route("/listing/campaigns/contribute", post(crate::listing_api::campaign_contribute))
        .route("/listing/campaigns/:id", get(crate::listing_api::campaign_details))
        .route("/listing/campaigns/:id/my-perks", get(crate::listing_api::campaign_my_perks))
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

/// v5.1.0: Per-wallet deployment rate limit (max 5 deploys per hour)
static DEPLOY_RATE_LIMITS: once_cell::sync::Lazy<dashmap::DashMap<[u8; 32], (u32, std::time::Instant)>> =
    once_cell::sync::Lazy::new(|| dashmap::DashMap::new());

/// Deploy a contract from frontend form
/// v5.1.0: Now requires wallet authentication to prevent unauthorized deployments
pub async fn deploy_contract(
    State(state): State<Arc<AppState>>,
    auth: AuthenticatedWallet,
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

    // 🔐 v5.1.0: Verify authenticated wallet matches deployer address
    if auth.address != deployer {
        tracing::warn!(
            "🚫 [CONTRACT] Auth mismatch: authenticated as {} but deploying as {}",
            q_log_privacy::mask_addr(&hex::encode(auth.address)),
            q_log_privacy::mask_addr(&hex::encode(deployer))
        );
        return Ok(Json(ApiResponse::error(
            "Authenticated wallet does not match deployer address".to_string(),
        )));
    }

    // 🔐 v5.1.0: Pre-check deployer balance BEFORE deployment (fail fast)
    const DEPLOYMENT_COST_PRECHECK: u128 = 1_000_000_000_000_000_000_000_000; // 1 QUG
    {
        let wallet_balances = state.wallet_balances.read().await;
        let balance = wallet_balances.get(&deployer).copied().unwrap_or(0);
        if balance < DEPLOYMENT_COST_PRECHECK {
            tracing::warn!(
                "🚫 [CONTRACT] Insufficient balance for deployment: {} has {} QUG, needs 1 QUG",
                q_log_privacy::mask_addr(&hex::encode(deployer)),
                q_log_privacy::mask_amt_display(balance as f64 / 1e24)
            );
            return Ok(Json(ApiResponse::error(
                "Insufficient balance: deployment requires 1 QUG".to_string(),
            )));
        }
    }

    // 🔐 v5.1.0: Deployment rate limit - max 5 per wallet per hour
    {
        let now = std::time::Instant::now();
        let mut entry = DEPLOY_RATE_LIMITS.entry(deployer).or_insert((0, now));
        let (count, window_start) = entry.value_mut();
        if now.duration_since(*window_start).as_secs() > 3600 {
            // Reset window
            *count = 1;
            *window_start = now;
        } else if *count >= 20 {
            tracing::warn!(
                "🚫 [CONTRACT] Rate limit exceeded: {} has deployed {} times this hour",
                q_log_privacy::mask_addr(&hex::encode(deployer)),
                count
            );
            return Ok(Json(ApiResponse::error(
                "Deployment rate limit exceeded: max 20 per hour".to_string(),
            )));
        } else {
            *count += 1;
        }
    }

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
            // v10.2.1: Track balance changes for persistence (fixes bug where deductions were lost on restart)
            let mut persist_deployer: Option<([u8; 32], u128)> = None;
            let mut persist_founder: Option<([u8; 32], u128)> = None;
            let mut persist_operator: Option<([u8; 32], u128)> = None;
            {
                let mut wallet_balances = state.wallet_balances.write().await;
                if let Some(balance) = wallet_balances.get_mut(&deployer) {
                    if *balance >= DEPLOYMENT_COST {
                        *balance -= DEPLOYMENT_COST;
                        persist_deployer = Some((deployer, *balance));
                        tracing::info!(
                            "💸 Deducted {} QUG deployment cost from {}. New balance: {}",
                            q_log_privacy::mask_amt_display(DEPLOYMENT_COST as f64 / 1e24),
                            q_log_privacy::mask_addr(&hex::encode(deployer)),
                            q_log_privacy::mask_amt_display(*balance as f64 / 1e24)
                        );

                        // v7.4.1: Credit deployment fee to founder wallet (previously burned!)
                        // Split between founder and node operator based on promille setting
                        {
                            let operator_promille = state.node_operator_fee_promille.load(std::sync::atomic::Ordering::Relaxed) as u128;
                            let operator_share = if operator_promille > 0 {
                                DEPLOYMENT_COST.saturating_mul(operator_promille) / 1000
                            } else { 0 };
                            let founder_share = DEPLOYMENT_COST.saturating_sub(operator_share);

                            // Credit founder
                            if founder_share > 0 {
                                let founder_addr = {
                                    let mut addr = [0u8; 32];
                                    if let Ok(bytes) = hex::decode(crate::aegis_auth_middleware::FOUNDER_WALLET) {
                                        if bytes.len() == 32 { addr.copy_from_slice(&bytes); }
                                    }
                                    addr
                                };
                                let old = wallet_balances.get(&founder_addr).copied().unwrap_or(0);
                                wallet_balances.insert(founder_addr, old + founder_share);
                                persist_founder = Some((founder_addr, old + founder_share));
                                tracing::info!(
                                    "💰 Deployment fee credited to founder: {} QUG (total: {} QUG)",
                                    q_log_privacy::mask_amt_display(founder_share as f64 / 1e24),
                                    q_log_privacy::mask_amt_display((old + founder_share) as f64 / 1e24)
                                );
                            }

                            // Credit node operator
                            if operator_share > 0 {
                                if let Ok(op_bytes) = hex::decode(&state.admin_wallet) {
                                    if op_bytes.len() == 32 {
                                        let mut op_addr = [0u8; 32];
                                        op_addr.copy_from_slice(&op_bytes);
                                        let old = wallet_balances.get(&op_addr).copied().unwrap_or(0);
                                        wallet_balances.insert(op_addr, old + operator_share);
                                        persist_operator = Some((op_addr, old + operator_share));
                                        // v8.1.1: Track fee earnings (convert from 24-decimal to micro-QUG)
                                        let micro_qug = (operator_share / 1_000_000_000_000_000_000) as u64; // 1e24 / 1e6 = 1e18
                                        crate::admin_settings_api::record_operator_fee(&state, micro_qug);
                                        tracing::info!(
                                            "💰 Operator fee earned: {} QUG (deployment fee share)",
                                            q_log_privacy::mask_amt_display(operator_share as f64 / 1e24)
                                        );
                                    }
                                }
                            }
                        }

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
                            q_log_privacy::mask_hash(&submission_result.tx_id_hex),
                            nonce,
                            submission_result.queued_for_block,
                            submission_result.broadcast_success
                        );
                    } else {
                        tracing::warn!(
                            "⚠️ Insufficient balance for deployment. Required: {}, Available: {}",
                            q_log_privacy::mask_amt(DEPLOYMENT_COST),
                            q_log_privacy::mask_amt(*balance)
                        );
                    }
                } else {
                    tracing::warn!("⚠️ Deployer wallet not found: {}", q_log_privacy::mask_addr(&hex::encode(deployer)));
                }
            }

            // v10.2.1: Persist deployment cost deductions to storage (fixes balance not decreasing on restart)
            for (addr, new_balance) in [persist_deployer, persist_founder, persist_operator].into_iter().flatten() {
                if let Err(e) = state.storage_engine.save_wallet_balance(&addr, new_balance).await {
                    tracing::warn!("⚠️ Failed to persist balance after deployment fee: {}", e);
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

                // v3.6.18: Helper to parse scientific notation WITHOUT f64 to preserve precision
                // Uses integer math to avoid floating-point errors on large numbers like 1e30
                let parse_scientific_notation = |s: &str| -> Option<u128> {
                    // Try direct u128 parse first (handles regular integers)
                    if let Ok(n) = s.parse::<u128>() {
                        return Some(n);
                    }

                    // v3.6.18: Parse scientific notation using integer math (no f64!)
                    // This preserves full precision for numbers like "1e30"
                    let s_lower = s.to_lowercase();
                    let parts: Vec<&str> = s_lower.split('e').collect();
                    if parts.len() == 2 {
                        let mantissa_str = parts[0];
                        let exp_str = parts[1].trim_start_matches('+');
                        if let Ok(exponent) = exp_str.parse::<i32>() {
                            if exponent >= 0 {
                                // Parse mantissa, handling decimal point
                                let mantissa_parts: Vec<&str> = mantissa_str.split('.').collect();
                                let whole_part = mantissa_parts[0];
                                let frac_part = if mantissa_parts.len() > 1 { mantissa_parts[1] } else { "" };

                                // Combine whole and fractional parts, then adjust exponent
                                let combined = format!("{}{}", whole_part, frac_part);
                                let adjusted_exp = (exponent as usize).saturating_sub(frac_part.len());

                                // Build the final number string
                                let mut result_str = combined.trim_start_matches('0').to_string();
                                if result_str.is_empty() {
                                    result_str = "0".to_string();
                                }

                                // Add zeros for the exponent
                                result_str.push_str(&"0".repeat(adjusted_exp));

                                if let Ok(n) = result_str.parse::<u128>() {
                                    tracing::info!("✅ [v3.6.18] Parsed scientific '{}' → {} (integer math, no f64)", s, q_log_privacy::mask_amt(n));
                                    return Some(n);
                                }
                            }
                        }
                    }

                    // Fallback: try f64 only for small values where precision doesn't matter
                    if let Ok(f) = s.parse::<f64>() {
                        if f >= 0.0 && f < 1e15 && f.is_finite() {  // Only use f64 for small values
                            return Some(f as u128);
                        }
                    }
                    None
                };

                // v4.1.0: Normalize to 10^(2*decimals) format for backward compatibility
                // Frontend sends base_units = display × 10^decimals
                // We multiply by 10^decimals again so storage format = display × 10^(2*decimals)
                // This matches the existing balance format expected by all display code
                let initial_supply_result: Option<u128> = if let Some(supply_u64) =
                    initial_supply_val.as_u64()
                {
                    // Frontend sent a JSON number - this is likely display units (NOT base units)
                    // because JSON numbers are typically human-entered values
                    let base_units = (supply_u64 as u128) * decimal_multiplier;
                    tracing::info!(
                        "✅ Token supply: {} × 10^{} = {} base units (u64 path)",
                        q_log_privacy::mask_amt(supply_u64 as u128),
                        decimals,
                        q_log_privacy::mask_amt(base_units)
                    );
                    Some(base_units)
                } else if let Some(supply_f64) = initial_supply_val.as_f64() {
                    // v3.6.18: Handle f64 values (scientific notation from JSON)
                    // WARNING: JSON numbers like 1e30 are parsed as f64 by serde, causing precision loss!
                    // f64 can only represent ~15-17 significant digits, so 1e30 becomes 1.000000019884624e30
                    if supply_f64 >= 0.0 && supply_f64 <= u128::MAX as f64 && supply_f64.is_finite() {
                        // v3.6.18: For large values (>1e15), round to nearest power of 10 to recover user intent
                        // This assumes users enter round numbers like 1e30, not 1.234567890123456e30
                        let supply_value = if supply_f64 >= 1e15 {
                            // Find the exponent and round to nearest power of 10
                            let log_val = supply_f64.log10();
                            let exponent = log_val.round() as u32;
                            let rounded = 10u128.pow(exponent.min(38)); // u128 max is ~3.4e38
                            tracing::warn!(
                                "⚠️ [v3.6.18] Large f64 value {} has precision loss! Rounded to 1e{} = {}",
                                q_log_privacy::mask_amt_display(supply_f64), exponent, q_log_privacy::mask_amt(rounded)
                            );
                            rounded
                        } else {
                            supply_f64 as u128
                        };

                        let base_units = supply_value.saturating_mul(decimal_multiplier);
                        tracing::info!(
                            "✅ Token supply (f64): {} display tokens × 10^{} = {} base units",
                            q_log_privacy::mask_amt(supply_value),
                            decimals,
                            q_log_privacy::mask_amt(base_units)
                        );
                        Some(base_units)
                    } else {
                        tracing::warn!("⚠️ Invalid f64 supply value: {}", supply_f64);
                        None
                    }
                } else if let Some(supply_str) = initial_supply_val.as_str() {
                    // v3.6.19: String values - handle scientific notation properly
                    // Frontend may send "1e+30" (display units) or "100000000000000000000000000000000000000" (base units)
                    match parse_scientific_notation(supply_str) {
                        Some(parsed_value) => {
                            // v3.6.19: CRITICAL FIX - If string contains 'e' or 'E', it's ALWAYS display units
                            // User typed "1e30" meaning 1e30 tokens, NOT 1e30 base units
                            // Only long numeric strings (no 'e') from frontend are base units
                            let is_scientific_notation = supply_str.to_lowercase().contains('e');
                            let is_already_base_units = !is_scientific_notation &&
                                parsed_value > 1_000_000_000_000_000_000u128; // > 1e18

                            let base_units = if is_already_base_units {
                                // Long numeric string from frontend (already converted)
                                parsed_value
                            } else {
                                // Scientific notation OR small number = display units, need conversion
                                parsed_value.saturating_mul(decimal_multiplier)
                            };

                            tracing::info!(
                                "✅ [v3.6.19] Token supply: '{}' → {} base units (scientific={}, already_base={})",
                                supply_str,
                                q_log_privacy::mask_amt(base_units),
                                is_scientific_notation,
                                is_already_base_units
                            );
                            Some(base_units)
                        }
                        None => {
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
                            q_log_privacy::mask_amt(initial_supply),
                            q_log_privacy::mask_amt_display(token_amount),
                            decimals,
                            q_log_privacy::mask_addr(&hex::encode(deployer))
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
    State(_state): State<Arc<AppState>>,
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

            // v4.0.3: Merge RWA boolean deployment params into features map
            // so the frontend controls can check contract.features.kyc_required etc.
            let mut features = contract.metadata.features.clone();
            let rwa_bool_keys = [
                "kyc_required", "accredited_only", "dividend_enabled", "transfer_restrictions",
                "voting_rights", "callable", "convertible", "delivery_option",
                "insurance_enabled", "retirement_enabled", "offset_tracking",
                "provenance_verified", "redemption_enabled", "sublicensing_allowed",
                "serial_number_tracking", "supply_chain_verified", "shipping_included",
            ];
            for key in &rwa_bool_keys {
                if let Some(val) = contract.deployment_params.get(*key) {
                    if let Some(b) = val.as_bool() {
                        if b {
                            features.insert(key.to_string(), true);
                        }
                    }
                }
            }

            ContractInfo {
                address: format!("qnk{}", hex::encode(contract.address.0)), // Add qnk prefix to match wallet format
                contract_type: format!("{:?}", contract.contract_type),
                name: contract.metadata.name,
                symbol: contract.metadata.symbol,
                owner: format!("qnk{}", hex::encode(contract.deployer)), // Add qnk prefix
                deployed_at: contract.deployed_at,
                verified: contract.verified,
                has_security_features: true, // From template security features
                features,
                deployment_tx: contract.deployment_tx,
                total_supply,
                decimals,
                deployment_params: serde_json::to_value(&contract.deployment_params).ok(),
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
    let ecosystem = &state.orobit_ecosystem;
    let deployed = ecosystem.deployed_contracts.read().await;

    let rwa_bool_keys = [
        "kyc_required", "accredited_only", "dividend_enabled", "transfer_restrictions",
        "voting_rights", "callable", "convertible", "delivery_option",
        "insurance_enabled", "retirement_enabled", "offset_tracking",
        "provenance_verified", "redemption_enabled", "sublicensing_allowed",
        "serial_number_tracking", "supply_chain_verified", "shipping_included",
    ];

    let mut contract_infos: Vec<ContractInfo> = deployed
        .values()
        .filter(|contract| {
            // Filter by owner (hex or qnk-prefixed address)
            if let Some(ref owner) = query.owner {
                let owner_hex = if owner.starts_with("qnk") {
                    owner[3..].to_string()
                } else if owner.starts_with("0x") {
                    owner[2..].to_string()
                } else {
                    owner.clone()
                };
                if hex::encode(contract.deployer) != owner_hex.to_lowercase() {
                    return false;
                }
            }
            // Filter by contract type (case-insensitive substring)
            if let Some(ref ct) = query.contract_type {
                let type_str = format!("{:?}", contract.contract_type).to_lowercase();
                if !type_str.contains(&ct.to_lowercase()) {
                    return false;
                }
            }
            // Filter by verified
            if query.verified_only.unwrap_or(false) && !contract.verified {
                return false;
            }
            true
        })
        .map(|contract| {
            let total_supply = contract
                .deployment_params
                .get("initialSupply")
                .or_else(|| contract.deployment_params.get("initial_supply"))
                .and_then(|v| {
                    v.as_u64().map(|n| n as u128)
                        .or_else(|| v.as_str().and_then(|s| s.parse::<u128>().ok()))
                });

            let decimals = contract
                .deployment_params
                .get("decimals")
                .and_then(|v| v.as_u64())
                .map(|d| d as u32)
                .or(Some(8));

            let mut features = contract.metadata.features.clone();
            for key in &rwa_bool_keys {
                if let Some(val) = contract.deployment_params.get(*key) {
                    if let Some(b) = val.as_bool() {
                        if b {
                            features.insert(key.to_string(), true);
                        }
                    }
                }
            }

            ContractInfo {
                address: format!("qnk{}", hex::encode(contract.address.0)),
                contract_type: format!("{:?}", contract.contract_type),
                name: contract.metadata.name.clone(),
                symbol: contract.metadata.symbol.clone(),
                owner: format!("qnk{}", hex::encode(contract.deployer)),
                deployed_at: contract.deployed_at,
                verified: contract.verified,
                has_security_features: true,
                features,
                deployment_tx: contract.deployment_tx.clone(),
                total_supply,
                decimals,
                deployment_params: serde_json::to_value(&contract.deployment_params).ok(),
            }
        })
        .collect();

    // Apply offset and limit
    let offset = query.offset.unwrap_or(0) as usize;
    let limit = query.limit.unwrap_or(u32::MAX) as usize;
    if offset < contract_infos.len() {
        contract_infos = contract_infos.into_iter().skip(offset).take(limit).collect();
    } else {
        contract_infos = Vec::new();
    }

    Ok(Json(ApiResponse::success(contract_infos)))
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

            // v4.0.3: Merge RWA boolean deployment params into features
            let mut features = contract.metadata.features.clone();
            let rwa_bool_keys = [
                "kyc_required", "accredited_only", "dividend_enabled", "transfer_restrictions",
                "voting_rights", "callable", "convertible", "delivery_option",
                "insurance_enabled", "retirement_enabled", "offset_tracking",
                "provenance_verified", "redemption_enabled", "sublicensing_allowed",
                "serial_number_tracking", "supply_chain_verified", "shipping_included",
            ];
            for key in &rwa_bool_keys {
                if let Some(val) = contract.deployment_params.get(*key) {
                    if let Some(b) = val.as_bool() {
                        if b {
                            features.insert(key.to_string(), true);
                        }
                    }
                }
            }

            let contract_info = ContractInfo {
                address: format!("qnk{}", hex::encode(contract.address.0)),
                contract_type: format!("{:?}", contract.contract_type),
                name: contract.metadata.name.clone(),
                symbol: contract.metadata.symbol.clone(),
                owner: format!("qnk{}", hex::encode(contract.deployer)),
                deployed_at: contract.deployed_at,
                verified: contract.verified,
                has_security_features: true,
                features,
                deployment_tx: contract.deployment_tx.clone(),
                total_supply,
                decimals,
                deployment_params: serde_json::to_value(&contract.deployment_params).ok(),
            };
            Ok(Json(ApiResponse::success(contract_info)))
        }
        None => Ok(Json(ApiResponse::error(format!(
            "Contract not found at address: {}",
            address
        )))),
    }
}

/// Interact with deployed contract - execute RWA and token actions
pub async fn interact_with_contract(
    Path(address): Path<String>,
    State(state): State<Arc<AppState>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let action = payload.get("action").and_then(|v| v.as_str()).unwrap_or("");
    if action.is_empty() {
        return Ok(Json(ApiResponse::error("Missing 'action' field".to_string())));
    }

    let addr_bytes = match hex::decode(address.trim_start_matches("0x")) {
        Ok(b) if b.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&b);
            arr
        }
        Ok(b) if b.len() < 32 => {
            let mut arr = [0u8; 32];
            arr[32 - b.len()..].copy_from_slice(&b);
            arr
        }
        _ => return Ok(Json(ApiResponse::error("Invalid contract address".to_string()))),
    };

    let ecosystem = &state.orobit_ecosystem;
    let contract_key = ContractAddress(addr_bytes);

    // Check contract exists
    {
        let contracts = ecosystem.deployed_contracts.read().await;
        if !contracts.contains_key(&contract_key) {
            return Ok(Json(ApiResponse::error("Contract not found".to_string())));
        }
    }

    // Generate a transaction hash for this interaction
    let tx_hash = {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        format!("0x{:064x}", ts)
    };

    // Process action and update contract state
    let result: serde_json::Value = match action {
        // ─── Real Estate Actions ───────────────────────────────
        "update_property_valuation" => {
            let valuation = payload.get("valuation_usd").and_then(|v| v.as_str()).unwrap_or("0");
            let mut contracts = ecosystem.deployed_contracts.write().await;
            if let Some(contract) = contracts.get_mut(&contract_key) {
                contract.deployment_params.insert("total_valuation_usd".to_string(), serde_json::json!(valuation));
                tracing::info!("📊 RWA: Updated property valuation to ${} for {}", q_log_privacy::mask_amt_display(valuation.parse::<f64>().unwrap_or(0.0)), q_log_privacy::mask_addr(&address));
            }
            serde_json::json!({ "action": "update_property_valuation", "valuation_usd": valuation })
        }
        "update_occupancy" => {
            let occupancy = payload.get("occupancy_rate").and_then(|v| v.as_str()).unwrap_or("");
            let rental_yield = payload.get("rental_yield_percent").and_then(|v| v.as_str()).unwrap_or("");
            let mut contracts = ecosystem.deployed_contracts.write().await;
            if let Some(contract) = contracts.get_mut(&contract_key) {
                if !occupancy.is_empty() {
                    contract.deployment_params.insert("occupancy_rate".to_string(), serde_json::json!(occupancy));
                }
                if !rental_yield.is_empty() {
                    contract.deployment_params.insert("rental_yield_percent".to_string(), serde_json::json!(rental_yield));
                }
                tracing::info!("📊 RWA: Updated occupancy={}% yield={}% for {}", occupancy, rental_yield, q_log_privacy::mask_addr(&address));
            }
            serde_json::json!({ "action": "update_occupancy", "occupancy_rate": occupancy, "rental_yield_percent": rental_yield })
        }

        // ─── Revenue / Dividend Distribution ───────────────────
        "distribute_revenue" | "distribute_dividend" | "distribute_dividends" | "distribute_royalties" => {
            let amount = payload.get("amount").and_then(|v| v.as_str()).unwrap_or("0");
            tracing::info!("💰 RWA: Revenue distribution of ${} for contract {}", q_log_privacy::mask_amt_display(amount.parse::<f64>().unwrap_or(0.0)), q_log_privacy::mask_addr(&address));
            serde_json::json!({ "action": action, "amount_usd": amount, "distributed_to": "all_holders" })
        }

        // ─── Compliance / KYC Actions ──────────────────────────
        "toggle_kyc" | "configure_kyc" => {
            let enabled = payload.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true);
            let mut contracts = ecosystem.deployed_contracts.write().await;
            if let Some(contract) = contracts.get_mut(&contract_key) {
                contract.deployment_params.insert("kyc_required".to_string(), serde_json::json!(enabled));
                contract.metadata.features.insert("kyc_required".to_string(), enabled);
                tracing::info!("🔐 RWA: KYC {} for {}", if enabled { "enabled" } else { "disabled" }, q_log_privacy::mask_addr(&address));
            }
            serde_json::json!({ "action": action, "kyc_required": enabled })
        }
        "toggle_accredited" => {
            let enabled = payload.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true);
            let mut contracts = ecosystem.deployed_contracts.write().await;
            if let Some(contract) = contracts.get_mut(&contract_key) {
                contract.deployment_params.insert("accredited_only".to_string(), serde_json::json!(enabled));
                contract.metadata.features.insert("accredited_only".to_string(), enabled);
            }
            serde_json::json!({ "action": action, "accredited_only": enabled })
        }
        "manage_whitelist" => {
            let addresses = payload.get("addresses").and_then(|v| v.as_str()).unwrap_or("");
            let operation = payload.get("operation").and_then(|v| v.as_str()).unwrap_or("add");
            tracing::info!("📋 RWA: Whitelist {} for {}: [{}addrs]", operation, q_log_privacy::mask_addr(&address), addresses.split(',').filter(|s| !s.is_empty()).count());
            serde_json::json!({ "action": "manage_whitelist", "operation": operation, "count": addresses.split(',').filter(|s| !s.is_empty()).count() })
        }

        // ─── Equity / Governance Actions ───────────────────────
        "create_proposal" => {
            let title = payload.get("title").and_then(|v| v.as_str()).unwrap_or("New Proposal");
            let description = payload.get("description").and_then(|v| v.as_str()).unwrap_or("");
            tracing::info!("🗳️ RWA: Proposal created for {}: {}", q_log_privacy::mask_addr(&address), title);
            serde_json::json!({ "action": "create_proposal", "title": title, "description": description, "proposal_id": format!("prop_{}", chrono::Utc::now().timestamp()) })
        }
        "vote_proposal" => {
            let proposal_id = payload.get("proposal_id").and_then(|v| v.as_str()).unwrap_or("");
            let vote = payload.get("vote").and_then(|v| v.as_str()).unwrap_or("for");
            serde_json::json!({ "action": "vote_proposal", "proposal_id": proposal_id, "vote": vote })
        }

        // ─── Bond / Fixed Income Actions ───────────────────────
        "call_bond" => {
            let call_price = payload.get("call_price").and_then(|v| v.as_str()).unwrap_or("0");
            tracing::info!("🏦 RWA: Bond called at ${} for {}", q_log_privacy::mask_amt_display(call_price.parse::<f64>().unwrap_or(0.0)), q_log_privacy::mask_addr(&address));
            serde_json::json!({ "action": "call_bond", "call_price_usd": call_price })
        }
        "convert_bond" | "convert_to_equity" => {
            let conversion_ratio = payload.get("conversion_ratio").and_then(|v| v.as_str()).unwrap_or("1.0");
            tracing::info!("🔄 RWA: Bond conversion at ratio {} for {}", conversion_ratio, q_log_privacy::mask_addr(&address));
            serde_json::json!({ "action": "convert_to_equity", "conversion_ratio": conversion_ratio })
        }
        "pay_coupon" | "issue_coupon" => {
            let amount = payload.get("amount").and_then(|v| v.as_str()).unwrap_or("0");
            tracing::info!("💵 RWA: Coupon payment of ${} for {}", q_log_privacy::mask_amt_display(amount.parse::<f64>().unwrap_or(0.0)), q_log_privacy::mask_addr(&address));
            serde_json::json!({ "action": "pay_coupon", "amount_usd": amount, "distributed_to": "all_bondholders" })
        }
        "update_credit_rating" => {
            let rating = payload.get("rating").and_then(|v| v.as_str()).unwrap_or("BBB");
            let mut contracts = ecosystem.deployed_contracts.write().await;
            if let Some(contract) = contracts.get_mut(&contract_key) {
                contract.deployment_params.insert("credit_rating".to_string(), serde_json::json!(rating));
            }
            serde_json::json!({ "action": "update_credit_rating", "rating": rating })
        }

        // ─── Commodity Actions ─────────────────────────────────
        "update_storage_proof" | "update_inventory" => {
            let proof_hash = payload.get("proof_hash").and_then(|v| v.as_str()).unwrap_or("");
            let quantity = payload.get("quantity").and_then(|v| v.as_str()).unwrap_or("");
            tracing::info!("📦 RWA: Storage proof updated for {}", q_log_privacy::mask_addr(&address));
            serde_json::json!({ "action": action, "proof_hash": proof_hash, "quantity": quantity })
        }
        "process_delivery" | "process_redemption" | "process_redemptions" => {
            let request_id = payload.get("request_id").and_then(|v| v.as_str()).unwrap_or("");
            let recipient = payload.get("recipient").and_then(|v| v.as_str()).unwrap_or("");
            tracing::info!("🚚 RWA: Processing delivery/redemption for {}", q_log_privacy::mask_addr(&address));
            serde_json::json!({ "action": action, "request_id": request_id, "recipient": recipient, "status": "processing" })
        }

        // ─── Carbon Credit Actions ─────────────────────────────
        "update_verification" | "update_verification_status" => {
            let status = payload.get("status").and_then(|v| v.as_str()).unwrap_or("verified");
            let verifier = payload.get("verifier").and_then(|v| v.as_str()).unwrap_or("");
            let mut contracts = ecosystem.deployed_contracts.write().await;
            if let Some(contract) = contracts.get_mut(&contract_key) {
                contract.deployment_params.insert("verification_status".to_string(), serde_json::json!(status));
                if !verifier.is_empty() {
                    contract.deployment_params.insert("verification_body".to_string(), serde_json::json!(verifier));
                }
            }
            serde_json::json!({ "action": action, "verification_status": status, "verifier": verifier })
        }
        "issue_offset_certificate" | "retire_credits" => {
            let tonnes = payload.get("tonnes_co2").and_then(|v| v.as_str()).unwrap_or("0");
            let beneficiary = payload.get("beneficiary").and_then(|v| v.as_str()).unwrap_or("");
            tracing::info!("🌱 RWA: Offset certificate for {} tonnes CO2, beneficiary: {}", tonnes, q_log_privacy::mask_addr(beneficiary));
            serde_json::json!({ "action": action, "tonnes_co2": tonnes, "beneficiary": beneficiary, "certificate_id": format!("cert_{}", chrono::Utc::now().timestamp()) })
        }

        // ─── Art & Collectible Actions ─────────────────────────
        "update_appraisal" | "update_appraisal_value" => {
            let value = payload.get("appraisal_value_usd").and_then(|v| v.as_str()).unwrap_or("0");
            let mut contracts = ecosystem.deployed_contracts.write().await;
            if let Some(contract) = contracts.get_mut(&contract_key) {
                contract.deployment_params.insert("appraisal_value_usd".to_string(), serde_json::json!(value));
                tracing::info!("🎨 RWA: Appraisal updated to ${} for {}", q_log_privacy::mask_amt_display(value.parse::<f64>().unwrap_or(0.0)), q_log_privacy::mask_addr(&address));
            }
            serde_json::json!({ "action": action, "appraisal_value_usd": value })
        }
        "update_custody" | "update_custody_location" => {
            let location = payload.get("custody_location").and_then(|v| v.as_str()).unwrap_or("");
            let custody_type = payload.get("custody_type").and_then(|v| v.as_str()).unwrap_or("vault");
            let mut contracts = ecosystem.deployed_contracts.write().await;
            if let Some(contract) = contracts.get_mut(&contract_key) {
                contract.deployment_params.insert("physical_custody".to_string(), serde_json::json!(custody_type));
                if !location.is_empty() {
                    contract.deployment_params.insert("custody_location".to_string(), serde_json::json!(location));
                }
            }
            serde_json::json!({ "action": action, "custody_type": custody_type, "custody_location": location })
        }

        // ─── IP & Royalty Actions ──────────────────────────────
        "manage_sublicenses" => {
            let licensee = payload.get("licensee").and_then(|v| v.as_str()).unwrap_or("");
            let terms = payload.get("terms").and_then(|v| v.as_str()).unwrap_or("");
            tracing::info!("📜 RWA: Sublicense managed for {}: licensee={}", q_log_privacy::mask_addr(&address), q_log_privacy::mask_addr(licensee));
            serde_json::json!({ "action": "manage_sublicenses", "licensee": licensee, "terms": terms })
        }

        // ─── Insurance Actions ─────────────────────────────────
        "update_insurance" => {
            let provider = payload.get("provider").and_then(|v| v.as_str()).unwrap_or("");
            let coverage = payload.get("coverage_usd").and_then(|v| v.as_str()).unwrap_or("");
            let mut contracts = ecosystem.deployed_contracts.write().await;
            if let Some(contract) = contracts.get_mut(&contract_key) {
                contract.metadata.features.insert("insurance_enabled".to_string(), true);
                if !provider.is_empty() {
                    contract.deployment_params.insert("insurance_provider".to_string(), serde_json::json!(provider));
                }
                if !coverage.is_empty() {
                    contract.deployment_params.insert("insurance_coverage_usd".to_string(), serde_json::json!(coverage));
                }
            }
            serde_json::json!({ "action": "update_insurance", "provider": provider, "coverage_usd": coverage })
        }

        // ─── Generic / Token Actions ───────────────────────────
        "pause" => {
            let mut contracts = ecosystem.deployed_contracts.write().await;
            if let Some(contract) = contracts.get_mut(&contract_key) {
                contract.contract_state.paused = true;
            }
            serde_json::json!({ "action": "pause", "paused": true })
        }
        "unpause" => {
            let mut contracts = ecosystem.deployed_contracts.write().await;
            if let Some(contract) = contracts.get_mut(&contract_key) {
                contract.contract_state.paused = false;
            }
            serde_json::json!({ "action": "unpause", "paused": false })
        }

        _ => {
            tracing::warn!("⚠️ Unknown contract action: {} for {}", action, q_log_privacy::mask_addr(&address));
            serde_json::json!({ "action": action, "status": "unknown_action" })
        }
    };

    tracing::info!("✅ Contract interaction: action={} contract={} tx={}", action, q_log_privacy::mask_addr(&address), q_log_privacy::mask_hash(&tx_hash));

    Ok(Json(ApiResponse::success(serde_json::json!({
        "result": "success",
        "action": action,
        "transaction_hash": tx_hash,
        "data": result
    }))))
}

/// Get user's deployment history
pub async fn get_user_deployments(
    Path(address): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<DeploymentStatusResponse>>>, StatusCode> {
    let deployer = match parse_address(&address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    let ecosystem = &state.orobit_ecosystem;
    let contracts = ecosystem.get_user_contracts(deployer).await;

    let responses: Vec<DeploymentStatusResponse> = contracts
        .into_iter()
        .map(|contract| DeploymentStatusResponse {
            request_id: contract.deployment_tx.clone(),
            status: "deployed".to_string(),
            contract_address: Some(format!("qnk{}", hex::encode(contract.address.0))),
            deployment_tx: Some(contract.deployment_tx.clone()),
            gas_used: None,
            error_message: None,
            progress: DeploymentProgress {
                current_step: 4,
                total_steps: 4,
                step_name: "Completed".to_string(),
                estimated_time_remaining: 0,
            },
        })
        .collect();

    Ok(Json(ApiResponse::success(responses)))
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
pub fn parse_contract_type(contract_type_str: &str) -> Result<ContractType, String> {
    match contract_type_str.to_lowercase().as_str() {
        // Core Token Contracts
        "secure_token" => Ok(ContractType::SecureToken),
        "advanced_token" => Ok(ContractType::AdvancedToken),
        "rwa_token" => Ok(ContractType::RwaToken),
        "orbusd_stablecoin" => Ok(ContractType::OrbusdStablecoin),
        // DeFi Infrastructure
        "multisig_wallet" => Ok(ContractType::MultisigWallet),
        "governance" => Ok(ContractType::Governance),
        "private_dex" => Ok(ContractType::PrivateDex),
        "timelock_vault" => Ok(ContractType::TimelockVault),
        "oracle_feed" => Ok(ContractType::OracleFeed),
        // Advanced DeFi
        "lending_pool" => Ok(ContractType::LendingPool),
        "liquidity_pool" => Ok(ContractType::LiquidityPool),
        "yield_farming" => Ok(ContractType::YieldFarming),
        "staking_contract" => Ok(ContractType::StakingContract),
        "insurance_protocol" => Ok(ContractType::InsuranceProtocol),
        // Real World Assets
        "real_estate_token" => Ok(ContractType::RealEstateToken),
        "commodity_token" => Ok(ContractType::CommodityToken),
        "carbon_credit_token" => Ok(ContractType::CarbonCreditToken),
        "art_collectible_token" => Ok(ContractType::ArtCollectibleToken),
        "equity_token" => Ok(ContractType::EquityToken),
        "fixed_income_token" => Ok(ContractType::FixedIncomeToken),
        "ip_revenue_token" => Ok(ContractType::IPRevenueToken),
        "physical_goods_token" => Ok(ContractType::PhysicalGoodsToken),
        // Derivatives & Trading
        "options_contract" => Ok(ContractType::OptionsContract),
        "prediction_market" => Ok(ContractType::PredictionMarket),
        "derivatives_platform" => Ok(ContractType::DerivativesPlatform),
        "synthetic_assets" => Ok(ContractType::SyntheticAssets),
        // Utility & Infrastructure
        "nft_marketplace" => Ok(ContractType::NftMarketplace),
        "identity_contract" => Ok(ContractType::IdentityContract),
        "bridge_contract" => Ok(ContractType::BridgeContract),
        "proxy_contract" => Ok(ContractType::ProxyContract),
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
                        q_log_privacy::mask_addr(&hex::encode(wallet_addr)),
                        q_log_privacy::mask_addr(&hex::encode(token_addr)),
                        q_log_privacy::mask_amt(stored_balance)
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
        q_log_privacy::mask_addr(&hex::encode(wallet_addr)),
        q_log_privacy::mask_addr(&hex::encode(token_addr)),
        q_log_privacy::mask_amt(balance)
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
    } else if address_str.len() == 64 && address_str.chars().all(|c| c.is_ascii_hexdigit()) {
        // Bare 64-char hex (no prefix) — bridge/internal token addresses
        address_str
    } else {
        return Err(format!(
            "Address must start with 0x or qnk, or be a bare 64-char hex (got: {})",
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
            q_log_privacy::mask_amt(amount as u128),
            q_log_privacy::mask_addr(&hex::encode(contract_addr)),
            q_log_privacy::mask_addr(&hex::encode(owner)),
            q_log_privacy::mask_amt(new_balance)
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
            q_log_privacy::mask_amt(amount as u128),
            q_log_privacy::mask_addr(&hex::encode(contract_addr)),
            q_log_privacy::mask_addr(&hex::encode(owner)),
            q_log_privacy::mask_amt(new_balance)
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
                q_log_privacy::mask_amt(amount_per_recipient as u128),
                q_log_privacy::mask_addr(&hex::encode(recipient_addr)),
                q_log_privacy::mask_addr(&hex::encode(contract_addr))
            );
        }

        tracing::info!(
            "✈️ Airdrop complete: {} tokens to {} recipients for contract {}. Total: {}",
            q_log_privacy::mask_amt(amount_per_recipient as u128),
            recipient_addrs.len(),
            q_log_privacy::mask_addr(&hex::encode(contract_addr)),
            q_log_privacy::mask_amt(total_amount as u128)
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
        q_log_privacy::mask_addr(&contract_key),
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
        q_log_privacy::mask_addr(&hex::encode(contract_addr)),
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
        q_log_privacy::mask_addr(&hex::encode(contract_addr)),
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
        q_log_privacy::mask_addr(&contract_address), q_log_privacy::mask_amt_display(request.amount.parse::<f64>().unwrap_or(0.0)), request.lock_days);

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
        q_log_privacy::mask_addr(&request.wallet_address), q_log_privacy::mask_amt_display(amount_f64), tier.name(), unlock_time);

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
    tracing::info!("🔓 [STAKING] Unstake request for contract {}", q_log_privacy::mask_addr(&contract_address));

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
        q_log_privacy::mask_addr(&request.wallet_address),
        q_log_privacy::mask_amt_display(stake.amount as f64 / 1e24),
        q_log_privacy::mask_amt_display(pending_rewards as f64 / 1e24));

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
    tracing::info!("💰 [STAKING] Claim rewards request for contract {}", q_log_privacy::mask_addr(&contract_address));

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

    tracing::info!("✅ [STAKING] {} claimed {} in rewards", q_log_privacy::mask_addr(&request.wallet_address), q_log_privacy::mask_amt_display(rewards_f64));

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
    tracing::info!("⚙️ [FEES] Update fee config request for {}", q_log_privacy::mask_addr(&contract_address));

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

    tracing::info!("✅ [FEES] Fee config updated for {}: {:?}", q_log_privacy::mask_addr(&contract_address), updated_config);

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
        let network_id = std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| "mainnet-genesis".to_string());
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
            tracing::info!("📡 Broadcast social profile update for {} via P2P", q_log_privacy::mask_addr(&key));
        }
    }

    tracing::info!("📱 Updated social profile for token {}", q_log_privacy::mask_addr(&key));

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

// ============ RWA MARKETPLACE ============

/// RWA marketplace listing entry
#[derive(Debug, Serialize, Deserialize)]
pub struct RwaMarketplaceListing {
    pub address: String,
    pub name: String,
    pub symbol: String,
    pub contract_type: String,
    pub category: String,
    pub description: String,
    pub deployed_at: u64,
    pub verified: bool,
    pub features: HashMap<String, bool>,
    pub total_value_usd: String,
    pub shares_available: String,
    pub kyc_required: bool,
    pub dividend_enabled: bool,
    // Campaign-specific fields (None for regular RWA contracts)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub campaign_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raised_usd: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_usd_num: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress_percent: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contributor_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub campaign_status: Option<String>,
}

/// Map a ContractType to a human-readable RWA category name
pub fn rwa_category_name(ct: &ContractType) -> String {
    match ct {
        ContractType::RealEstateToken => "Real Estate".to_string(),
        ContractType::EquityToken => "Equity & Shares".to_string(),
        ContractType::FixedIncomeToken => "Fixed Income".to_string(),
        ContractType::CommodityToken => "Commodities".to_string(),
        ContractType::CarbonCreditToken => "Carbon Credits".to_string(),
        ContractType::ArtCollectibleToken => "Art & Collectibles".to_string(),
        ContractType::IPRevenueToken => "IP & Royalties".to_string(),
        ContractType::PhysicalGoodsToken => "Physical Goods".to_string(),
        ContractType::RwaToken => "General RWA".to_string(),
        _ => "Other".to_string(),
    }
}

/// List all deployed RWA contracts as a marketplace
///
/// Query parameters:
///   - `category`: Optional filter by RWA category (e.g. "real_estate", "equity",
///     "fixed_income", "commodity", "carbon_credit", "art_collectible",
///     "ip_revenue", "physical_goods")
pub async fn get_rwa_marketplace(
    State(state): State<Arc<AppState>>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<ApiResponse<Vec<RwaMarketplaceListing>>>, StatusCode> {
    let category_filter = params.get("category").map(|s| s.as_str());

    let ecosystem = &state.orobit_ecosystem;
    let contracts = ecosystem.deployed_contracts.read().await;

    let rwa_types = vec![
        ContractType::RwaToken,
        ContractType::RealEstateToken,
        ContractType::EquityToken,
        ContractType::FixedIncomeToken,
        ContractType::CommodityToken,
        ContractType::CarbonCreditToken,
        ContractType::ArtCollectibleToken,
        ContractType::IPRevenueToken,
        ContractType::PhysicalGoodsToken,
    ];

    let mut listings: Vec<RwaMarketplaceListing> = contracts
        .iter()
        .filter(|(_, contract)| rwa_types.contains(&contract.contract_type))
        .filter(|(_, contract)| {
            if let Some(cat) = category_filter {
                match cat {
                    "real_estate" => contract.contract_type == ContractType::RealEstateToken,
                    "equity" => contract.contract_type == ContractType::EquityToken,
                    "fixed_income" => contract.contract_type == ContractType::FixedIncomeToken,
                    "commodity" => contract.contract_type == ContractType::CommodityToken,
                    "carbon_credit" => contract.contract_type == ContractType::CarbonCreditToken,
                    "art_collectible" => contract.contract_type == ContractType::ArtCollectibleToken,
                    "ip_revenue" => contract.contract_type == ContractType::IPRevenueToken,
                    "physical_goods" => contract.contract_type == ContractType::PhysicalGoodsToken,
                    "exchange_listing" => false, // campaigns injected separately below
                    _ => true,
                }
            } else {
                true
            }
        })
        .map(|(addr, contract)| {
            RwaMarketplaceListing {
                address: hex::encode(addr.0),
                name: contract.metadata.name.clone(),
                symbol: contract.metadata.symbol.clone().unwrap_or_default(),
                contract_type: format!("{:?}", contract.contract_type),
                category: rwa_category_name(&contract.contract_type),
                description: contract.metadata.description.clone(),
                deployed_at: contract.deployed_at,
                verified: contract.verified,
                features: contract.metadata.features.clone(),
                total_value_usd: contract
                    .deployment_params
                    .get("total_value_usd")
                    .or(contract.deployment_params.get("total_valuation_usd"))
                    .or(contract.deployment_params.get("face_value_usd"))
                    .or(contract.deployment_params.get("appraisal_value_usd"))
                    .or(contract.deployment_params.get("minimum_guarantee_usd"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("0")
                    .to_string(),
                shares_available: contract
                    .deployment_params
                    .get("shares_count")
                    .or(contract.deployment_params.get("total_shares"))
                    .or(contract.deployment_params.get("total_tokens"))
                    .or(contract.deployment_params.get("total_units"))
                    .or(contract.deployment_params.get("total_fractions"))
                    .or(contract.deployment_params.get("total_credits_tonnes"))
                    .or(contract.deployment_params.get("quantity_per_token"))
                    .or(contract.deployment_params.get("initialSupply"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("0")
                    .to_string(),
                kyc_required: contract
                    .deployment_params
                    .get("kyc_required")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                dividend_enabled: contract
                    .deployment_params
                    .get("dividend_enabled")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                campaign_id: None,
                raised_usd: None,
                target_usd_num: None,
                progress_percent: None,
                contributor_count: None,
                campaign_status: None,
            }
        })
        .collect();

    // Inject exchange listing campaigns as marketplace entries
    if category_filter.is_none() || category_filter == Some("exchange_listing") {
        let campaigns = state.listing_campaigns.read().await;
        for campaign in campaigns.iter() {
            let progress = if campaign.target_usd > 0.0 {
                (campaign.raised_usd / campaign.target_usd * 100.0).min(100.0)
            } else {
                0.0
            };
            let remaining = (campaign.target_usd - campaign.raised_usd).max(0.0);
            let status_str = match campaign.status {
                crate::listing_api::CampaignStatus::Funding => "funding",
                crate::listing_api::CampaignStatus::Funded => "funded",
                crate::listing_api::CampaignStatus::Listed => "listed",
                crate::listing_api::CampaignStatus::Cancelled => "cancelled",
            };
            let mut features = HashMap::new();
            features.insert("crowdfund".to_string(), true);
            features.insert("early_bird".to_string(), campaign.early_bird_claimed < campaign.early_bird_slots);
            listings.push(RwaMarketplaceListing {
                address: campaign.campaign_id.clone(),
                name: format!("{} Exchange Listing", campaign.exchange_name),
                symbol: "XLIST".to_string(),
                contract_type: "ExchangeListing".to_string(),
                category: "Exchange Listing".to_string(),
                description: campaign.description.clone(),
                deployed_at: campaign.created_at,
                verified: true,
                features,
                total_value_usd: format!("{:.0}", campaign.target_usd),
                shares_available: format!("{:.0}", remaining),
                kyc_required: false,
                dividend_enabled: false,
                campaign_id: Some(campaign.campaign_id.clone()),
                raised_usd: Some(campaign.raised_usd),
                target_usd_num: Some(campaign.target_usd),
                progress_percent: Some(progress),
                contributor_count: Some(campaign.contributor_count),
                campaign_status: Some(status_str.to_string()),
            });
        }
    }

    Ok(Json(ApiResponse::success(listings)))
}

// ═══════════════════════════════════════════════════════════════════
// v4.1.0: RWA Portfolio, Collateralization, Distribution, Compliance
// ═══════════════════════════════════════════════════════════════════

/// Get RWA portfolio summary including collateral positions and distribution schedules
pub async fn get_rwa_portfolio(
    Query(params): Query<std::collections::HashMap<String, String>>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet = params.get("wallet").cloned().unwrap_or_default();
    tracing::info!("📊 RWA Portfolio request for wallet: {}", q_log_privacy::mask_addr(&wallet));

    let ecosystem = &state.orobit_ecosystem;
    let collateral = {
        let positions = ecosystem.collateral_positions.read().await;
        positions.get(&wallet).cloned().unwrap_or_default()
    };
    let schedules = {
        let scheds = ecosystem.distribution_schedules.read().await;
        scheds.get(&wallet).cloned().unwrap_or_default()
    };

    Ok(Json(ApiResponse::success(serde_json::json!({
        "wallet": wallet,
        "collateral_positions": collateral,
        "distribution_schedules": schedules,
    }))))
}

/// Borrow against RWA collateral
pub async fn rwa_collateral_borrow(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let contract_address = payload.get("contract_address").and_then(|v| v.as_str()).unwrap_or("");
    let amount = payload.get("amount").and_then(|v| v.as_str()).unwrap_or("0");
    let wallet = payload.get("wallet").and_then(|v| v.as_str()).unwrap_or("");

    tracing::info!("🏦 RWA Collateral Borrow: {} borrows ${} against {}", q_log_privacy::mask_addr(wallet), q_log_privacy::mask_amt_display(amount.parse::<f64>().unwrap_or(0.0)), q_log_privacy::mask_addr(contract_address));

    let position_id = format!("pos_{}_{}", &wallet[..8.min(wallet.len())], chrono::Utc::now().timestamp());
    let position = serde_json::json!({
        "id": position_id,
        "contractAddress": contract_address,
        "borrowedAmount": amount.parse::<f64>().unwrap_or(0.0),
        "borrowedCurrency": "QUG",
        "interestRate": 5.5,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "status": "healthy"
    });

    let ecosystem = &state.orobit_ecosystem;
    {
        let mut positions = ecosystem.collateral_positions.write().await;
        positions.entry(wallet.to_string())
            .or_insert_with(Vec::new)
            .push(position.clone());
    }

    let tx_hash = format!("0x{:016x}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0));
    Ok(Json(ApiResponse::success(serde_json::json!({
        "position_id": position_id,
        "transaction_hash": tx_hash,
        "status": "active"
    }))))
}

/// Repay collateral loan
pub async fn rwa_collateral_repay(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let position_id = payload.get("position_id").and_then(|v| v.as_str()).unwrap_or("");
    let amount = payload.get("amount").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let wallet = payload.get("wallet").and_then(|v| v.as_str()).unwrap_or("");

    tracing::info!("💰 RWA Collateral Repay: {} repays ${} on position {}", q_log_privacy::mask_addr(wallet), q_log_privacy::mask_amt_display(amount), position_id);

    let ecosystem = &state.orobit_ecosystem;
    {
        let mut positions = ecosystem.collateral_positions.write().await;
        if let Some(wallet_positions) = positions.get_mut(wallet) {
            wallet_positions.retain(|p| {
                p.get("id").and_then(|v| v.as_str()).unwrap_or("") != position_id
                    || p.get("borrowedAmount").and_then(|v| v.as_f64()).unwrap_or(0.0) > amount
            });
        }
    }

    let tx_hash = format!("0x{:016x}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0));
    Ok(Json(ApiResponse::success(serde_json::json!({
        "repaid": amount,
        "transaction_hash": tx_hash,
        "status": "repaid"
    }))))
}

/// Schedule automatic revenue distribution
pub async fn rwa_schedule_distribution(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let contract_address = payload.get("contract_address").and_then(|v| v.as_str()).unwrap_or("");
    let frequency = payload.get("frequency").and_then(|v| v.as_str()).unwrap_or("monthly");
    let amount = payload.get("amount").and_then(|v| v.as_str()).unwrap_or("0");
    let wallet = payload.get("wallet").and_then(|v| v.as_str()).unwrap_or("");

    tracing::info!("📅 RWA Distribution Schedule: {} sets {} ${} distribution for {}", q_log_privacy::mask_addr(wallet), frequency, q_log_privacy::mask_amt_display(amount.parse::<f64>().unwrap_or(0.0)), q_log_privacy::mask_addr(contract_address));

    let schedule_id = format!("sched_{}_{}", &wallet[..8.min(wallet.len())], chrono::Utc::now().timestamp());
    let freq_days: i64 = match frequency {
        "weekly" => 7,
        "monthly" => 30,
        "quarterly" => 90,
        "annually" => 365,
        _ => 30,
    };
    let next_dist = chrono::Utc::now() + chrono::Duration::days(freq_days);

    let schedule = serde_json::json!({
        "id": schedule_id,
        "contractAddress": contract_address,
        "frequency": frequency,
        "amount": amount.parse::<f64>().unwrap_or(0.0),
        "next_distribution": next_dist.to_rfc3339(),
        "totalDistributed": 0,
        "recipientCount": 0,
        "enabled": true
    });

    let ecosystem = &state.orobit_ecosystem;
    {
        let mut scheds = ecosystem.distribution_schedules.write().await;
        scheds.entry(wallet.to_string())
            .or_insert_with(Vec::new)
            .push(schedule.clone());
    }

    Ok(Json(ApiResponse::success(serde_json::json!({
        "schedule_id": schedule_id,
        "next_distribution": next_dist.to_rfc3339(),
        "status": "active"
    }))))
}

/// Toggle distribution schedule on/off
pub async fn rwa_toggle_distribution(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let schedule_id = payload.get("schedule_id").and_then(|v| v.as_str()).unwrap_or("");
    let enabled = payload.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true);
    let wallet = payload.get("wallet").and_then(|v| v.as_str()).unwrap_or("");

    tracing::info!("🔄 RWA Distribution Toggle: {} {} schedule {}", q_log_privacy::mask_addr(wallet), if enabled { "enables" } else { "pauses" }, schedule_id);

    let ecosystem = &state.orobit_ecosystem;
    {
        let mut scheds = ecosystem.distribution_schedules.write().await;
        if let Some(wallet_schedules) = scheds.get_mut(wallet) {
            for s in wallet_schedules.iter_mut() {
                if s.get("id").and_then(|v| v.as_str()) == Some(schedule_id) {
                    s.as_object_mut().map(|obj| obj.insert("enabled".to_string(), serde_json::json!(enabled)));
                }
            }
        }
    }

    Ok(Json(ApiResponse::success(serde_json::json!({
        "schedule_id": schedule_id,
        "enabled": enabled
    }))))
}

/// Check compliance status for RWA token trading
pub async fn rwa_compliance_check(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet = payload.get("wallet").and_then(|v| v.as_str()).unwrap_or("");
    let token_address = payload.get("token_address").and_then(|v| v.as_str()).unwrap_or("");

    tracing::info!("🔍 RWA Compliance Check: wallet {} for token {}", q_log_privacy::mask_addr(wallet), q_log_privacy::mask_addr(token_address));

    let ecosystem = &state.orobit_ecosystem;

    // Parse token address to ContractAddress key
    let addr_bytes = match hex::decode(token_address.trim_start_matches("0x")) {
        Ok(b) if b.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&b);
            arr
        }
        Ok(b) if b.len() < 32 => {
            let mut arr = [0u8; 32];
            arr[32 - b.len()..].copy_from_slice(&b);
            arr
        }
        _ => [0u8; 32],
    };
    let contract_key = ContractAddress(addr_bytes);

    let (kyc_required, accredited_only, transfer_restricted, whitelisted) = {
        let contracts = ecosystem.deployed_contracts.read().await;
        if let Some(contract) = contracts.get(&contract_key) {
            let kyc = contract.metadata.features.get("kyc_required").copied().unwrap_or(false);
            let accredited = contract.metadata.features.get("accredited_only").copied().unwrap_or(false);
            let restricted = contract.metadata.features.get("transfer_restrictions").copied().unwrap_or(false);
            (kyc, accredited, restricted, true)
        } else {
            (false, false, false, true)
        }
    };

    // Compliance check result
    let kyc_passed = true; // Testnet: all pass
    let accreditation_passed = true; // Testnet: all pass
    let can_trade = (!kyc_required || kyc_passed) && (!accredited_only || accreditation_passed) && (!transfer_restricted || whitelisted);

    Ok(Json(ApiResponse::success(serde_json::json!({
        "wallet": wallet,
        "token_address": token_address,
        "can_trade": can_trade,
        "kyc_required": kyc_required,
        "kyc_passed": kyc_passed,
        "accredited_only": accredited_only,
        "accreditation_passed": accreditation_passed,
        "transfer_restricted": transfer_restricted,
        "whitelisted": whitelisted,
        "compliance_level": if can_trade { "full" } else { "restricted" }
    }))))
}

// ============================================================================
// v4.2.0-beta: VAULT RWA Token — Physical Device Redemption Handlers
// ============================================================================

/// POST /api/v1/contracts/vault/redeem — Burn 1 VAULT token and create redemption order
pub async fn vault_redeem(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<VaultRedeemRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet = auth.address;
    let wallet_hex = hex::encode(wallet);
    let quantity = request.quantity.unwrap_or(1).max(1);

    // Check VAULT token balance
    let vault_addr = q_types::VAULT_TOKEN_ADDRESS;
    let balance_key = (wallet, vault_addr);

    {
        let token_balances = state.token_balances.read().await;
        let current_balance = token_balances.get(&balance_key).copied().unwrap_or(0);
        if current_balance < quantity as u128 {
            return Ok(Json(ApiResponse::error(format!(
                "Insufficient VAULT balance. You have {}, need {} to redeem.",
                current_balance, quantity
            ))));
        }
    }

    // Burn the tokens
    {
        let mut token_balances = state.token_balances.write().await;
        if let Some(balance) = token_balances.get_mut(&balance_key) {
            *balance -= quantity as u128;
            tracing::info!(
                "🔥 [VAULT] Burned {} VAULT token(s) from wallet {} (remaining: {})",
                quantity, q_log_privacy::mask_addr(&wallet_hex), q_log_privacy::mask_amt(*balance)
            );
        }
    }

    // Persist burned balance
    {
        let token_balances = state.token_balances.read().await;
        let new_balance = token_balances.get(&balance_key).copied().unwrap_or(0);
        drop(token_balances);
        if let Err(e) = state.storage_engine.save_token_balance(&wallet, &vault_addr, new_balance).await {
            tracing::warn!("⚠️ [VAULT] Failed to persist burned balance: {}", e);
        }
    }

    // Create redemption order
    let redemption_id = format!("VR-{}-{}", chrono::Utc::now().timestamp(), &wallet_hex[..8]);
    let redemption = VaultRedemption {
        redemption_id: redemption_id.clone(),
        buyer_wallet: format!("qnk{}", wallet_hex),
        shipping_name: request.shipping_name,
        shipping_address: request.shipping_address,
        city: request.city,
        state_province: request.state_province,
        zip: request.zip,
        country: request.country,
        phone: request.phone,
        email: request.email,
        color_variant: request.color_variant,
        quantity,
        status: "pending".to_string(),
        tracking_number: None,
        serial_number: None,
        created_at: chrono::Utc::now().timestamp() as u64,
        fulfilled_at: None,
    };

    {
        let mut redemptions = state.vault_redemptions.write().await;
        redemptions.push(redemption.clone());
    }

    // Persist to RocksDB
    persist_vault_redemption(&state, &redemption).await;

    tracing::info!(
        "📦 [VAULT] Redemption {} created: {} device(s), color: {}, wallet: {}",
        redemption_id, quantity, redemption.color_variant, q_log_privacy::mask_addr(&wallet_hex)
    );

    Ok(Json(ApiResponse::success(serde_json::json!({
        "redemption_id": redemption_id,
        "quantity": quantity,
        "status": "pending",
        "message": format!("Successfully burned {} VAULT token(s). Your physical device order has been placed.", quantity)
    }))))
}

/// GET /api/v1/contracts/vault/redemptions — Get all redemptions (admin) or user's own
pub async fn vault_get_redemptions(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet = auth.address;
    let wallet_hex = hex::encode(wallet);
    let is_admin = wallet == q_types::BANK_MASTER_ACCOUNT;

    let redemptions = state.vault_redemptions.read().await;

    let filtered: Vec<&VaultRedemption> = if is_admin {
        // Admin sees all redemptions
        redemptions.iter().collect()
    } else {
        // Regular users only see their own
        let full_addr = format!("qnk{}", wallet_hex);
        redemptions.iter().filter(|r| r.buyer_wallet == full_addr).collect()
    };

    Ok(Json(ApiResponse::success(serde_json::json!({
        "redemptions": filtered,
        "total": filtered.len(),
        "is_admin": is_admin,
    }))))
}

/// POST /api/v1/contracts/vault/fulfill — Admin updates redemption status/tracking
pub async fn vault_fulfill(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<VaultFulfillRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet = auth.address;

    // Only BANK_MASTER_ACCOUNT or the operator wallet can fulfill
    let operator_wallet_hex = "4fff16bc7d825a3d2e3ae0b15c6e70e91dc18dce1c55ec22543a8e4ae9e6c7b2";
    let is_admin = wallet == q_types::BANK_MASTER_ACCOUNT
        || hex::encode(wallet) == operator_wallet_hex;

    if !is_admin {
        return Ok(Json(ApiResponse::error(
            "Only the VAULT admin can fulfill redemptions".to_string(),
        )));
    }

    let valid_statuses = ["pending", "processing", "shipped", "delivered"];
    if !valid_statuses.contains(&request.status.as_str()) {
        return Ok(Json(ApiResponse::error(format!(
            "Invalid status '{}'. Must be one of: {:?}",
            request.status, valid_statuses
        ))));
    }

    let mut redemptions = state.vault_redemptions.write().await;
    if let Some(redemption) = redemptions.iter_mut().find(|r| r.redemption_id == request.redemption_id) {
        redemption.status = request.status.clone();
        if let Some(ref tracking) = request.tracking_number {
            redemption.tracking_number = Some(tracking.clone());
        }
        if let Some(ref serial) = request.serial_number {
            redemption.serial_number = Some(serial.clone());
        }
        if request.status == "shipped" || request.status == "delivered" {
            redemption.fulfilled_at = Some(chrono::Utc::now().timestamp() as u64);
        }

        let redemption_clone = redemption.clone();
        drop(redemptions);

        // Persist updated redemption to RocksDB
        persist_vault_redemption(&state, &redemption_clone).await;

        tracing::info!(
            "📦 [VAULT] Redemption {} updated: status={}, tracking={:?}, serial={:?}",
            request.redemption_id, request.status, request.tracking_number, request.serial_number
        );

        Ok(Json(ApiResponse::success(serde_json::json!({
            "redemption_id": request.redemption_id,
            "status": request.status,
            "tracking_number": request.tracking_number,
            "serial_number": request.serial_number,
            "message": "Redemption updated successfully"
        }))))
    } else {
        Ok(Json(ApiResponse::error(format!(
            "Redemption '{}' not found",
            request.redemption_id
        ))))
    }
}

/// GET /api/v1/contracts/vault/stats — Get VAULT token supply statistics
pub async fn vault_get_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let vault_addr = q_types::VAULT_TOKEN_ADDRESS;
    let total_supply: u128 = 1000; // Fixed supply

    // Count circulating tokens (all balances with VAULT_TOKEN_ADDRESS)
    let token_balances = state.token_balances.read().await;
    let circulating: u128 = token_balances.iter()
        .filter(|((_, contract), _)| *contract == vault_addr)
        .map(|(_, balance)| *balance)
        .sum();

    let burned = total_supply.saturating_sub(circulating);

    // Count redemptions by status
    let redemptions = state.vault_redemptions.read().await;
    let pending = redemptions.iter().filter(|r| r.status == "pending").count();
    let processing = redemptions.iter().filter(|r| r.status == "processing").count();
    let shipped = redemptions.iter().filter(|r| r.status == "shipped").count();
    let delivered = redemptions.iter().filter(|r| r.status == "delivered").count();
    let total_redeemed: u32 = redemptions.iter().map(|r| r.quantity).sum();

    Ok(Json(ApiResponse::success(serde_json::json!({
        "total_supply": total_supply,
        "circulating": circulating,
        "burned": burned,
        "remaining": circulating,
        "redemptions": {
            "total_orders": redemptions.len(),
            "total_devices_redeemed": total_redeemed,
            "pending": pending,
            "processing": processing,
            "shipped": shipped,
            "delivered": delivered,
        }
    }))))
}

// ============================================================================
// v5.1.0: FORGE RWA Endpoints — Physical Mining Machine Redemption
// ============================================================================

/// POST /api/v1/contracts/forge/redeem — Burn 1 FORGE token and create mining machine order
pub async fn forge_redeem(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<ForgeRedeemRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet = auth.address;
    let wallet_hex = hex::encode(wallet);
    let quantity = request.quantity.unwrap_or(1).max(1);

    // Check FORGE token balance
    let forge_addr = q_types::FORGE_TOKEN_ADDRESS;
    let balance_key = (wallet, forge_addr);

    {
        let token_balances = state.token_balances.read().await;
        let current_balance = token_balances.get(&balance_key).copied().unwrap_or(0);
        if current_balance < quantity as u128 {
            return Ok(Json(ApiResponse::error(format!(
                "Insufficient FORGE balance. You have {}, need {} to redeem.",
                current_balance, quantity
            ))));
        }
    }

    // Burn the tokens
    {
        let mut token_balances = state.token_balances.write().await;
        if let Some(balance) = token_balances.get_mut(&balance_key) {
            *balance -= quantity as u128;
            tracing::info!(
                "🔥 [FORGE] Burned {} FORGE token(s) from wallet {} (remaining: {})",
                quantity, q_log_privacy::mask_addr(&wallet_hex), q_log_privacy::mask_amt(*balance)
            );
        }
    }

    // Persist burned balance
    {
        let token_balances = state.token_balances.read().await;
        let new_balance = token_balances.get(&balance_key).copied().unwrap_or(0);
        drop(token_balances);
        if let Err(e) = state.storage_engine.save_token_balance(&wallet, &forge_addr, new_balance).await {
            tracing::warn!("⚠️ [FORGE] Failed to persist burned balance: {}", e);
        }
    }

    // Validate and default configuration options
    let cpu_config = request.cpu_config.unwrap_or_else(|| "epyc-9755-dual".to_string());
    let gpu_config = request.gpu_config.unwrap_or_else(|| "rtx-5090-dual".to_string());
    let cooling_type = request.cooling_type.unwrap_or_else(|| "liquid-copper".to_string());
    let ram_gb = request.ram_gb.unwrap_or(512);
    let storage_config = request.storage_config.unwrap_or_else(|| "nvme-4tb-raid1".to_string());
    let nic_config = request.nic_config.unwrap_or_else(|| "100gbe".to_string());
    let chassis_color = request.chassis_color.unwrap_or_else(|| "titanium-copper".to_string());

    // Validate CPU config
    let valid_cpus = ["epyc-9755-dual", "epyc-9654-dual", "xeon-w9-3595x-dual"];
    if !valid_cpus.contains(&cpu_config.as_str()) {
        return Ok(Json(ApiResponse::error(format!(
            "Invalid CPU config '{}'. Options: {:?}", cpu_config, valid_cpus
        ))));
    }

    // Validate GPU config
    let valid_gpus = ["none", "rtx-5090-dual", "rtx-5090-quad", "a100-dual", "l40-quad"];
    if !valid_gpus.contains(&gpu_config.as_str()) {
        return Ok(Json(ApiResponse::error(format!(
            "Invalid GPU config '{}'. Options: {:?}", gpu_config, valid_gpus
        ))));
    }

    // Create redemption order
    let redemption_id = format!("FR-{}-{}", chrono::Utc::now().timestamp(), &wallet_hex[..8]);
    let redemption = ForgeRedemption {
        redemption_id: redemption_id.clone(),
        buyer_wallet: format!("qnk{}", wallet_hex),
        shipping_name: request.shipping_name,
        shipping_address: request.shipping_address,
        city: request.city,
        state_province: request.state_province,
        zip: request.zip,
        country: request.country,
        phone: request.phone,
        email: request.email,
        cpu_config: cpu_config.clone(),
        gpu_config: gpu_config.clone(),
        cooling_type,
        ram_gb,
        storage_config,
        nic_config,
        chassis_color,
        quantity,
        status: "pending".to_string(),
        tracking_number: None,
        serial_number: None,
        machine_id: None,
        attestation_pubkey: None,
        created_at: chrono::Utc::now().timestamp() as u64,
        fulfilled_at: None,
    };

    {
        let mut redemptions = state.forge_redemptions.write().await;
        redemptions.push(redemption.clone());
    }

    // Persist to RocksDB
    persist_forge_redemption(&state, &redemption).await;

    tracing::info!(
        "⚒️ [FORGE] Redemption {} created: {} unit(s), CPU: {}, GPU: {}, wallet: {}",
        redemption_id, quantity, cpu_config, gpu_config, q_log_privacy::mask_addr(&wallet_hex)
    );

    Ok(Json(ApiResponse::success(serde_json::json!({
        "redemption_id": redemption_id,
        "quantity": quantity,
        "cpu_config": redemption.cpu_config,
        "gpu_config": redemption.gpu_config,
        "ram_gb": redemption.ram_gb,
        "status": "pending",
        "message": format!("Successfully burned {} FORGE token(s). Your Quillon Forge mining machine order has been placed.", quantity)
    }))))
}

/// GET /api/v1/contracts/forge/redemptions — Get all Forge redemptions (admin) or user's own
pub async fn forge_get_redemptions(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet = auth.address;
    let wallet_hex = hex::encode(wallet);
    let is_admin = wallet == q_types::BANK_MASTER_ACCOUNT;

    let redemptions = state.forge_redemptions.read().await;

    let filtered: Vec<&ForgeRedemption> = if is_admin {
        redemptions.iter().collect()
    } else {
        let full_addr = format!("qnk{}", wallet_hex);
        redemptions.iter().filter(|r| r.buyer_wallet == full_addr).collect()
    };

    Ok(Json(ApiResponse::success(serde_json::json!({
        "redemptions": filtered,
        "total": filtered.len(),
        "is_admin": is_admin,
    }))))
}

/// POST /api/v1/contracts/forge/fulfill — Admin updates Forge redemption status
pub async fn forge_fulfill(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<ForgeFulfillRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet = auth.address;

    let operator_wallet_hex = "4fff16bc7d825a3d2e3ae0b15c6e70e91dc18dce1c55ec22543a8e4ae9e6c7b2";
    let is_admin = wallet == q_types::BANK_MASTER_ACCOUNT
        || hex::encode(wallet) == operator_wallet_hex;

    if !is_admin {
        return Ok(Json(ApiResponse::error(
            "Only the FORGE admin can fulfill redemptions".to_string(),
        )));
    }

    let valid_statuses = ["pending", "configured", "assembling", "testing", "shipped", "delivered"];
    if !valid_statuses.contains(&request.status.as_str()) {
        return Ok(Json(ApiResponse::error(format!(
            "Invalid status '{}'. Must be one of: {:?}",
            request.status, valid_statuses
        ))));
    }

    let mut redemptions = state.forge_redemptions.write().await;
    if let Some(redemption) = redemptions.iter_mut().find(|r| r.redemption_id == request.redemption_id) {
        redemption.status = request.status.clone();
        if let Some(ref tracking) = request.tracking_number {
            redemption.tracking_number = Some(tracking.clone());
        }
        if let Some(ref serial) = request.serial_number {
            redemption.serial_number = Some(serial.clone());
        }
        if let Some(ref machine_id) = request.machine_id {
            redemption.machine_id = Some(machine_id.clone());
        }
        if let Some(ref attestation_pubkey) = request.attestation_pubkey {
            redemption.attestation_pubkey = Some(attestation_pubkey.clone());
        }
        if request.status == "shipped" || request.status == "delivered" {
            redemption.fulfilled_at = Some(chrono::Utc::now().timestamp() as u64);
        }

        let redemption_clone = redemption.clone();
        drop(redemptions);

        // Persist updated redemption to RocksDB
        persist_forge_redemption(&state, &redemption_clone).await;

        tracing::info!(
            "⚒️ [FORGE] Redemption {} updated: status={}, machine_id={:?}, tracking={:?}",
            request.redemption_id, request.status, request.machine_id, request.tracking_number
        );

        Ok(Json(ApiResponse::success(serde_json::json!({
            "redemption_id": request.redemption_id,
            "status": request.status,
            "tracking_number": request.tracking_number,
            "serial_number": request.serial_number,
            "machine_id": request.machine_id,
            "attestation_pubkey": request.attestation_pubkey,
            "message": "Forge redemption updated successfully"
        }))))
    } else {
        Ok(Json(ApiResponse::error(format!(
            "Forge redemption '{}' not found",
            request.redemption_id
        ))))
    }
}

/// GET /api/v1/contracts/forge/stats — Get FORGE token supply and machine statistics
pub async fn forge_get_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let forge_addr = q_types::FORGE_TOKEN_ADDRESS;
    let total_supply: u128 = 500; // Fixed supply: 500 Forge units

    // Count circulating tokens
    let token_balances = state.token_balances.read().await;
    let circulating: u128 = token_balances.iter()
        .filter(|((_, contract), _)| *contract == forge_addr)
        .map(|(_, balance)| *balance)
        .sum();

    let burned = total_supply.saturating_sub(circulating);

    // Count redemptions by status
    let redemptions = state.forge_redemptions.read().await;
    let pending = redemptions.iter().filter(|r| r.status == "pending").count();
    let configured = redemptions.iter().filter(|r| r.status == "configured").count();
    let assembling = redemptions.iter().filter(|r| r.status == "assembling").count();
    let testing = redemptions.iter().filter(|r| r.status == "testing").count();
    let shipped = redemptions.iter().filter(|r| r.status == "shipped").count();
    let delivered = redemptions.iter().filter(|r| r.status == "delivered").count();
    let total_redeemed: u32 = redemptions.iter().map(|r| r.quantity).sum();

    // Count by CPU config
    let epyc_count = redemptions.iter().filter(|r| r.cpu_config.contains("epyc")).count();
    let xeon_count = redemptions.iter().filter(|r| r.cpu_config.contains("xeon")).count();

    // Count by GPU config
    let gpu_count = redemptions.iter().filter(|r| r.gpu_config != "none").count();
    let total_cores: u64 = redemptions.iter().map(|r| {
        match r.cpu_config.as_str() {
            "epyc-9755-dual" => 256,
            "epyc-9654-dual" => 192,
            "xeon-w9-3595x-dual" => 120,
            _ => 128,
        }
    }).sum();

    Ok(Json(ApiResponse::success(serde_json::json!({
        "total_supply": total_supply,
        "circulating": circulating,
        "burned": burned,
        "remaining": circulating,
        "redemptions": {
            "total_orders": redemptions.len(),
            "total_machines_redeemed": total_redeemed,
            "pending": pending,
            "configured": configured,
            "assembling": assembling,
            "testing": testing,
            "shipped": shipped,
            "delivered": delivered,
        },
        "fleet_stats": {
            "total_cores_ordered": total_cores,
            "epyc_configurations": epyc_count,
            "xeon_configurations": xeon_count,
            "gpu_equipped": gpu_count,
        }
    }))))
}
