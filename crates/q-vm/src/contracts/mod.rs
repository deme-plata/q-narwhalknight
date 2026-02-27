/// Contracts module - Q-NarwhalKnight VM Smart Contract System
///
/// This module provides the complete smart contract infrastructure for the
/// Q-NarwhalKnight VM, including Orobit Chimera contract integration.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod orobit_smart_contracts;
pub mod security;
pub mod collateral_vault;
pub mod bio_contracts;
/// v3.7.4: Post-quantum contract deployment signatures (Dilithium5)
pub mod pq_deployment;
/// v3.9.1: Decentralized identity with inheritance support
pub mod decentralized_identity;
/// v8.5.5: Quillon Credit yield vault (lock QUG → mint QCREDIT, tiered APY)
pub mod qcredit_vault;

// Re-export main types for convenience
pub use orobit_smart_contracts::{
    ContractAddress, ContractMetadata, ContractState, ContractType, DeployedSmartContract,
    DeploymentOptions, FormDefinition, OrobitSmartContractEcosystem, SmartContractTemplate,
};

// Re-export security types
pub use security::{
    AccessControl, AuditStatus, Pausable, PullPayment, ReentrancyGuard, Roles, SafeMath, SecurityAnalyzer,
    SecurityConfig, SecurityReport, SecuritySuite,
};

// Re-export collateral vault types
pub use collateral_vault::{
    CollateralVault, LiquidationResult, MintResult, PositionHealth, RedeemResult, VaultStats,
    LIQUIDATION_BONUS, LIQUIDATION_RATIO, MIN_COLLATERAL_RATIO, WARNING_RATIO,
};

// Re-export post-quantum deployment types (v3.7.4)
pub use pq_deployment::{
    ContractDeployerKeys, PQContractDeployment, PQDeploymentRequest,
};

// Re-export bio contract types
pub use bio_contracts::{
    // License system
    BioLicense, BioLicenseContract, DEASchedule, LicenseType,
    // Synthesis proof
    SynthesisProof, SynthesisProofContract, VerificationMethod,
    // BioToken
    BioTokenContract, StakeInfo, StakingTier,
    // Safety oracle
    BioSafetyOracleContract, OracleInfo, SafetyClassification, SafetyVote,
    // Marketplace
    SynthesisMarketplaceContract, SynthesisListing, SynthesisOrder, OrderStatus,
    // Errors
    BioContractError,
};

// Re-export decentralized identity types (v3.9.1)
pub use decentralized_identity::{
    DecentralizedIdentityContract, OnChainIdentity, OnChainDeathCertificate,
    InheritanceTransfer, KycLevel, TokenType as IdentityTokenType,
    IdentityEvent, IdentityError, IdentityContractStats,
    IdentityContractMethod, IdentityContractResult,
};

// Re-export QCredit vault types (v8.5.5)
pub use qcredit_vault::{
    QCreditVault, CreditPosition, CreditTier, TierInfo, VaultStatus as QCreditVaultStatus,
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
    /// Create a new ContractRegistry with an existing Orobit ecosystem
    ///
    /// This is the preferred constructor as it allows sharing the ecosystem
    /// instance (with storage) across the application.
    pub fn new_with_ecosystem(ecosystem: std::sync::Arc<OrobitSmartContractEcosystem>) -> Self {
        Self {
            contracts: std::sync::RwLock::new(HashMap::new()),
            orobit_ecosystem: ecosystem,
        }
    }

    /// Create a new ContractRegistry with a fresh ecosystem (NO STORAGE)
    ///
    /// ⚠️ DEPRECATED: This creates an ecosystem WITHOUT persistent storage.
    /// Use `new_with_ecosystem()` instead to share the storage-backed ecosystem.
    #[deprecated(note = "Use new_with_ecosystem() to share storage-backed ecosystem")]
    pub async fn new() -> anyhow::Result<Self> {
        eprintln!("⚠️⚠️⚠️ WARNING: ContractRegistry::new() creates ecosystem WITHOUT storage!");
        eprintln!("⚠️⚠️⚠️ Use new_with_ecosystem() instead for persistence!");
        panic!("DEPRECATED: ContractRegistry::new() should not be called! Use new_with_ecosystem() instead.");
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
        let (contract_id, _address) = self.orobit_ecosystem
            .deploy_contract(contract_type, deployer, parameters, options)
            .await?;
        Ok(contract_id)
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

    /// Iterate over all deployed contracts
    ///
    /// Returns a vector of (address, contract) tuples for all deployed contracts.
    /// This is useful for enumerating contracts for DEX token discovery, analytics, etc.
    pub fn iter_contracts(&self) -> Vec<([u8; 32], std::sync::Arc<Contract>)> {
        let contracts = self.contracts.read().unwrap();
        contracts.iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect()
    }

    /// Get count of deployed contracts
    pub fn contract_count(&self) -> usize {
        let contracts = self.contracts.read().unwrap();
        contracts.len()
    }
}
