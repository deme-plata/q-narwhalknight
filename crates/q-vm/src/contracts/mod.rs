/// Contracts module - Q-NarwhalKnight VM Smart Contract System
///
/// This module provides the complete smart contract infrastructure for the
/// Q-NarwhalKnight VM, including Orobit Chimera contract integration.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod orobit_smart_contracts;
pub mod security;

// Re-export main types for convenience
pub use orobit_smart_contracts::{
    ContractAddress, ContractType, DeployedSmartContract, DeploymentOptions, FormDefinition,
    OrobitSmartContractEcosystem, SmartContractTemplate,
};

// Re-export security types
pub use security::{
    AccessControl, AuditStatus, Pausable, PullPayment, ReentrancyGuard, Roles, SafeMath, SecurityAnalyzer,
    SecurityConfig, SecurityReport, SecuritySuite,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractCall {
    pub contract_address: [u8; 32],
    pub method: String,
    pub args: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardingCapability {
    None,
    DataParallel,
    ModelParallel,
    Horizontal,
    Vertical,
    Full,
}

#[derive(Debug, Clone)]
pub struct AIModelCall {
    pub model_id: String,
    pub input: Vec<u8>,
    pub model: String,
    pub shard_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub min_cpu_cores: u32,
    pub min_memory_mb: u64,
    pub min_gpu_memory_mb: u64,
    pub preferred_batch_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistration {
    pub model_id: String,
    pub version: String,
    pub owner: [u8; 32],
    pub description: String,
    pub capabilities: ShardingCapability,
    pub resources: ResourceRequirements,
    pub hash: [u8; 32],
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct Contract {
    pub code: Vec<u8>,
    pub state: HashMap<Vec<u8>, Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct ContractResult {
    pub success: bool,
    pub return_data: Vec<u8>,
    pub error: Option<String>,
    pub gas_used: u64,
    pub state_changes: HashMap<Vec<u8>, Vec<u8>>,
    pub logs: Vec<String>,
}

pub struct ContractRegistry {
    contracts: std::sync::RwLock<HashMap<[u8; 32], std::sync::Arc<Contract>>>,
    orobit_ecosystem: std::sync::Arc<OrobitSmartContractEcosystem>,
}

impl ContractRegistry {
    pub async fn new() -> anyhow::Result<Self> {
        Ok(Self {
            contracts: std::sync::RwLock::new(HashMap::new()),
            orobit_ecosystem: std::sync::Arc::new(OrobitSmartContractEcosystem::new().await?),
        })
    }

    pub fn get(&self, address: &[u8; 32]) -> Option<std::sync::Arc<Contract>> {
        let contracts = self.contracts.read().unwrap();
        contracts.get(address).cloned()
    }

    /// Get Orobit smart contract ecosystem
    pub fn get_orobit_ecosystem(&self) -> std::sync::Arc<OrobitSmartContractEcosystem> {
        self.orobit_ecosystem.clone()
    }

    /// Deploy Orobit smart contract
    pub async fn deploy_orobit_contract(
        &self,
        contract_type: ContractType,
        deployer: [u8; 32],
        parameters: HashMap<String, serde_json::Value>,
        options: DeploymentOptions,
    ) -> anyhow::Result<String> {
        self.orobit_ecosystem
            .deploy_contract(contract_type, deployer, parameters, options)
            .await
    }

    /// Get available Orobit contract templates
    pub async fn get_available_orobit_contracts(&self) -> Vec<ContractType> {
        self.orobit_ecosystem.get_available_contracts().await
    }

    /// Get deployment form for Orobit contract
    pub async fn get_orobit_deployment_form(
        &self,
        contract_type: &ContractType,
    ) -> anyhow::Result<FormDefinition> {
        self.orobit_ecosystem
            .get_form_definition(contract_type)
            .await
    }
}
