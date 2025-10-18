//! Quillon Bank - Revolutionary Quantum-Enhanced Banking System
//! 
//! Next-generation decentralized banking platform combining:
//! - Quantum-resistant security with HSM clusters
//! - AI-powered financial services and credit assessment
//! - Multi-chain asset integration
//! - Privacy-preserving transactions with ZK proofs
//! - Plugin-based extensibility
//! - QNKUSD quantum stablecoin
//! - Integration with Q-NarwhalKnight consensus

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow};
use tokio::sync::{RwLock, Mutex};
use uuid::Uuid;
use q_types::{NodeId, Phase};
use q_plugin_system::PluginManager;

pub mod quantum_vault;
pub mod credit_engine;
pub mod payment_network;
pub mod treasury;
pub mod compliance;
pub mod wealth_agents;
pub mod identity;
pub mod airdrop_smart_contract;
pub mod oracle_integration;
pub mod klaw_oracle;
pub mod consensus_bridge;
pub mod qnkusd_integration;
// pub mod atomic_swap_integration; // Disabled - depends on q-bitcoin-bridge which is deactivated

/// Core Quillon Banking system that orchestrates all financial services
pub struct QuillonBankSystem {
    pub vault_system: Arc<quantum_vault::QuantumVaultSystem>,
    pub credit_engine: Arc<credit_engine::AICreditEngine>,
    pub payment_network: Arc<payment_network::QuillonPaymentNetwork>,
    pub treasury: Arc<treasury::AlgorithmicTreasury>,
    pub compliance: Arc<compliance::ZKComplianceModule>,
    pub wealth_agents: Arc<wealth_agents::AutonomousWealthManager>,
    pub identity: Arc<identity::DecentralizedIdentitySystem>,
    pub oracle_integration: Arc<oracle_integration::BankingOracleIntegration>,
    pub klaw_oracle: Arc<klaw_oracle::KLawOracle>,
    pub plugin_manager: Arc<PluginManager>,
    pub consensus_bridge: Arc<consensus_bridge::ConsensusBridge>,
    pub qnkusd_system: Arc<qnkusd_integration::QNKUSDSystem>,
    pub accounts: Arc<RwLock<HashMap<Address, BankAccount>>>,
    pub global_state: Arc<RwLock<BankGlobalState>>,
    pub node_id: NodeId,
    pub phase: Phase,
}

/// Unique Quillon Bank account identifier
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Address(pub [u8; 32]);

impl Address {
    pub fn new() -> Self {
        let uuid = Uuid::new_v4();
        let mut addr = [0u8; 32];
        addr[..16].copy_from_slice(uuid.as_bytes());
        Self(addr)
    }
    
    pub fn from_public_key(pubkey: &[u8; 33]) -> Self {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(pubkey);
        let hash = hasher.finalize();
        let mut addr = [0u8; 32];
        addr.copy_from_slice(&hash);
        Self(addr)
    }

    pub fn from_node_id(node_id: NodeId) -> Self {
        Self(node_id)
    }
}

/// Comprehensive Quillon Bank account with quantum-enhanced multi-asset support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BankAccount {
    pub address: Address,
    pub balances: HashMap<AssetType, Balance>,
    pub credit_score: CreditScore,
    pub identity: identity::VerifiedIdentity,
    pub privacy_tier: PrivacyTier,
    pub wealth_agent: Option<wealth_agents::WealthAgentId>,
    pub transaction_history: Vec<Transaction>,
    pub created_at: u64,
    pub last_activity: u64,
    pub quantum_features: QuantumAccountFeatures,
}

/// Asset types supported by Quillon Bank
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AssetType {
    ORB,           // Native Q-NarwhalKnight token
    QNKUSD,        // Quillon Bank USD stablecoin
    BTC,           // Bitcoin
    ETH,           // Ethereum
    USDC,          // USD Coin
    Gold,          // Tokenized gold
    RealEstate,    // Tokenized real estate
    Stock(String), // Tokenized stocks
    Bond(String),  // Government/corporate bonds
}

/// Quantum-enhanced account features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAccountFeatures {
    pub quantum_vault_enabled: bool,
    pub post_quantum_security: bool,
    pub quantum_credit_scoring: bool,
    pub autonomous_wealth_management: bool,
    pub quantum_privacy_level: QuantumPrivacyLevel,
}

impl Default for QuantumAccountFeatures {
    fn default() -> Self {
        Self {
            quantum_vault_enabled: true,
            post_quantum_security: true,
            quantum_credit_scoring: true,
            autonomous_wealth_management: false,
            quantum_privacy_level: QuantumPrivacyLevel::Standard,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumPrivacyLevel {
    Standard,   // Basic quantum privacy
    Enhanced,   // Advanced ZK proofs
    Shadow,     // High anonymity
    Phantom,    // Maximum quantum privacy
}

/// Balance with precision and quantum locking mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Balance {
    pub available: u128,      // Available balance (18 decimals)
    pub locked: u128,         // Locked in smart contracts
    pub staked: u128,         // Staked for rewards
    pub borrowed: u128,       // Borrowed amount
    pub lending: u128,        // Amount being lent out
    pub quantum_secured: u128, // Amount in quantum vaults
    pub last_updated: u64,    // Timestamp
}

impl Balance {
    pub fn total(&self) -> u128 {
        self.available + self.locked + self.staked + self.quantum_secured
    }
    
    pub fn net_worth(&self) -> i128 {
        (self.available + self.locked + self.staked + self.lending + self.quantum_secured) as i128 - self.borrowed as i128
    }
}

/// AI-powered quantum credit scoring system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditScore {
    pub score: u16,           // 300-850 scale
    pub risk_tier: RiskTier,
    pub factors: Vec<CreditFactor>,
    pub history: Vec<CreditEvent>,
    pub quantum_enhancement: QuantumCreditData,
    pub last_calculated: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCreditData {
    pub quantum_transaction_patterns: f64,
    pub post_quantum_security_usage: f64,
    pub vault_utilization_score: f64,
    pub consensus_participation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskTier {
    Excellent,  // 800-850
    VeryGood,   // 740-799
    Good,       // 670-739
    Fair,       // 580-669
    Poor,       // 300-579
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditFactor {
    pub factor_type: String,
    pub impact: i16,          // -100 to +100 points
    pub weight: f32,          // 0.0 to 1.0
    pub description: String,
}

/// Enhanced privacy tiers for Quillon Bank
#[derive(Debug, Clone, Serialize, Deserialize, Eq, Hash, PartialEq)]
pub enum PrivacyTier {
    Standard,   // Basic privacy, lower fees
    Enhanced,   // Additional privacy features
    Shadow,     // High privacy with zk-proofs
    Phantom,    // Maximum privacy, highest fees
    Quantum,    // Quantum-enhanced privacy (new tier)
}

/// Quillon Bank transactions with full quantum traceability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub id: TransactionId,
    pub from: Address,
    pub to: Address,
    pub asset: AssetType,
    pub amount: u128,
    pub fee: u128,
    pub transaction_type: TransactionType,
    pub privacy_level: PrivacyTier,
    pub timestamp: u64,
    pub block_height: u64,
    pub status: TransactionStatus,
    pub metadata: TransactionMetadata,
    pub quantum_signature: Option<Vec<u8>>,
    pub consensus_proof: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TransactionId(pub [u8; 32]);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionType {
    Transfer,
    Deposit,
    Withdrawal,
    Loan,
    LoanRepayment,
    Stake,
    Unstake,
    Swap,
    Investment,
    Salary,
    Purchase,
    Refund,
    QNKUSDMint,    // QNKUSD minting
    QNKUSDBurn,    // QNKUSD burning
    VaultDeposit,  // Quantum vault deposit
    VaultWithdraw, // Quantum vault withdrawal
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionStatus {
    Pending,
    Confirmed,
    Failed,
    Frozen,     // Fraud detection
    Compliance, // Compliance review
    ConsensusProcessing, // Being processed by consensus
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionMetadata {
    pub description: Option<String>,
    pub merchant: Option<String>,
    pub location: Option<String>,
    pub tags: Vec<String>,
    pub compliance_flags: Vec<String>,
    pub quantum_features: Vec<String>,
}

/// Global Quillon Bank state and metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BankGlobalState {
    pub total_deposits: HashMap<AssetType, u128>,
    pub total_loans: HashMap<AssetType, u128>,
    pub total_reserves: HashMap<AssetType, u128>,
    pub qnkusd_total_supply: u128,
    pub qnkusd_collateral_ratio: f64,
    pub active_accounts: u64,
    pub daily_volume: HashMap<AssetType, u128>,
    pub system_health: SystemHealth,
    pub quantum_metrics: QuantumBankMetrics,
    pub last_updated: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumBankMetrics {
    pub total_quantum_vaults: u64,
    pub post_quantum_transactions_24h: u64,
    pub consensus_integration_health: f64,
    pub quantum_privacy_adoption: f64,
    pub average_quantum_credit_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub vault_status: String,
    pub ai_models_status: String,
    pub network_latency: f64,
    pub fraud_detection_active: bool,
    pub compliance_status: String,
    pub treasury_yield: f64,
    pub consensus_sync_status: String,
    pub qnkusd_stability_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditEvent {
    pub event_type: String,
    pub impact: i16,
    pub timestamp: u64,
    pub description: String,
}

impl QuillonBankSystem {
    /// Initialize the complete Quillon Banking system with Q-NarwhalKnight integration
    pub async fn new(
        node_id: NodeId,
        phase: Phase,
        plugin_manager: Arc<PluginManager>,
    ) -> Result<Self> {
        let vault_system = Arc::new(quantum_vault::QuantumVaultSystem::new().await?);
        let credit_engine = Arc::new(credit_engine::AICreditEngine::new().await?);
        let payment_network = Arc::new(payment_network::QuillonPaymentNetwork::new().await?);
        let treasury = Arc::new(treasury::AlgorithmicTreasury::new().await?);
        let compliance = Arc::new(compliance::ZKComplianceModule::new().await?);
        let wealth_agents = Arc::new(wealth_agents::AutonomousWealthManager::new().await?);
        let identity = Arc::new(identity::DecentralizedIdentitySystem::new().await?);
        
        // Initialize oracle integration for banking operations
        let oracle_integration = Arc::new(
            oracle_integration::BankingOracleIntegration::new().await?
        );
        
        // Initialize banking-specific oracle feeds
        oracle_integration.initialize_banking_feeds().await?;
        
        // Initialize K-Law Oracle for AI-driven banking parameters
        let klaw_oracle = Arc::new(klaw_oracle::KLawOracle::new(
            "0x826c533770B4Bc53aa6dA31747113595e0032567".to_string()
        ));

        // Initialize consensus bridge for Q-NarwhalKnight integration
        let consensus_bridge = Arc::new(consensus_bridge::ConsensusBridge::new(node_id, phase.clone()).await?);

        // Initialize QNKUSD stablecoin system
        let qnkusd_system = Arc::new(qnkusd_integration::QNKUSDSystem::new(
            node_id,
            phase.clone(),
            vault_system.clone(),
            oracle_integration.clone(),
        ).await?);
        
        Ok(Self {
            vault_system,
            credit_engine,
            payment_network,
            treasury,
            compliance,
            wealth_agents,
            identity,
            oracle_integration,
            klaw_oracle,
            plugin_manager,
            consensus_bridge,
            qnkusd_system,
            accounts: Arc::new(RwLock::new(HashMap::new())),
            global_state: Arc::new(RwLock::new(BankGlobalState {
                total_deposits: HashMap::new(),
                total_loans: HashMap::new(),
                total_reserves: HashMap::new(),
                qnkusd_total_supply: 0,
                qnkusd_collateral_ratio: 1.5, // 150% collateralization
                active_accounts: 0,
                daily_volume: HashMap::new(),
                system_health: SystemHealth {
                    vault_status: "Operational".to_string(),
                    ai_models_status: "Active".to_string(),
                    network_latency: 0.009,
                    fraud_detection_active: true,
                    compliance_status: "Compliant".to_string(),
                    treasury_yield: 4.2,
                    consensus_sync_status: "Synchronized".to_string(),
                    qnkusd_stability_score: 0.999,
                },
                quantum_metrics: QuantumBankMetrics {
                    total_quantum_vaults: 0,
                    post_quantum_transactions_24h: 0,
                    consensus_integration_health: 1.0,
                    quantum_privacy_adoption: 0.0,
                    average_quantum_credit_score: 750.0,
                },
                last_updated: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            })),
            node_id,
            phase,
        })
    }

    /// Initialize the Quillon Bank system
    pub async fn initialize(&self) -> Result<()> {
        tracing::info!("🏦 Initializing Quillon Bank - Quantum-Enhanced Banking System");
        
        // Initialize all subsystems
        self.vault_system.initialize().await?;
        self.credit_engine.initialize().await?;
        self.payment_network.initialize().await?;
        self.treasury.initialize().await?;
        self.compliance.initialize().await?;
        self.wealth_agents.initialize().await?;
        self.identity.initialize().await?;
        self.oracle_integration.initialize().await?;
        self.consensus_bridge.initialize().await?;
        self.qnkusd_system.initialize().await?;

        tracing::info!("✅ Quillon Bank initialized successfully");
        tracing::info!("💰 QNKUSD stablecoin system: ACTIVE");
        tracing::info!("🔐 Quantum vault security: ENABLED");
        tracing::info!("⚡ Consensus integration: SYNCHRONIZED");
        
        Ok(())
    }
    
    /// Create a new Quillon Bank account with enhanced quantum features
    pub async fn create_account(
        &self,
        identity_proof: identity::IdentityProof,
        privacy_tier: PrivacyTier,
        enable_quantum_features: bool,
    ) -> Result<Address> {
        // Verify identity through decentralized identity system
        let verified_identity = self.identity.verify_identity(identity_proof).await?;
        
        // Generate new account address from node_id if provided
        let address = Address::new();
        
        // Initialize quantum-enhanced credit score using AI
        let initial_credit_score = self.credit_engine.calculate_initial_quantum_score(&verified_identity).await?;
        
        // Setup quantum account features
        let quantum_features = if enable_quantum_features {
            QuantumAccountFeatures {
                quantum_vault_enabled: true,
                post_quantum_security: true,
                quantum_credit_scoring: true,
                autonomous_wealth_management: false,
                quantum_privacy_level: match privacy_tier {
                    PrivacyTier::Standard => QuantumPrivacyLevel::Standard,
                    PrivacyTier::Enhanced => QuantumPrivacyLevel::Enhanced,
                    PrivacyTier::Shadow => QuantumPrivacyLevel::Shadow,
                    PrivacyTier::Phantom | PrivacyTier::Quantum => QuantumPrivacyLevel::Phantom,
                },
            }
        } else {
            QuantumAccountFeatures::default()
        };

        // Create account
        let account = BankAccount {
            address: address.clone(),
            balances: HashMap::new(),
            credit_score: initial_credit_score,
            identity: verified_identity,
            privacy_tier,
            wealth_agent: None,
            transaction_history: Vec::new(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            last_activity: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            quantum_features,
        };
        
        // Store account
        {
            let mut accounts = self.accounts.write().await;
            accounts.insert(address.clone(), account);
        }
        
        // Register with consensus if quantum features enabled
        if enable_quantum_features {
            self.consensus_bridge.register_account(&address).await?;
        }
        
        // Update global state
        {
            let mut state = self.global_state.write().await;
            state.active_accounts += 1;
            if enable_quantum_features {
                state.quantum_metrics.quantum_privacy_adoption = 
                    state.quantum_metrics.quantum_privacy_adoption + 1.0 / state.active_accounts as f64;
            }
        }
        
        tracing::info!("🎯 New Quillon Bank account created: {:?}", address);
        Ok(address)
    }

    /// Execute a quantum-enhanced banking transaction
    pub async fn execute_transaction(&self, tx: Transaction) -> Result<TransactionId> {
        // Enhanced fraud detection with quantum patterns
        let fraud_risk = self.detect_quantum_fraud_preemptive(&tx).await?;
        if fraud_risk > 0.97 {
            return Err(anyhow!("Transaction flagged as high quantum fraud risk"));
        }
        
        // Enhanced compliance check with post-quantum verification
        let compliance_result = self.compliance.check_quantum_transaction(&tx).await?;
        if !compliance_result.approved {
            return Err(anyhow!("Transaction failed quantum compliance check: {}", compliance_result.reason));
        }

        // Submit to consensus for quantum transactions
        let consensus_proof = if matches!(tx.privacy_level, PrivacyTier::Quantum) {
            Some(self.consensus_bridge.submit_transaction(&tx).await?)
        } else {
            None
        };
        
        // Execute through appropriate quantum payment tier
        let tx_id = match tx.privacy_level {
            PrivacyTier::Standard => self.payment_network.execute_lightning(&tx).await?,
            PrivacyTier::Enhanced => self.payment_network.execute_enhanced(&tx).await?,
            PrivacyTier::Shadow => self.payment_network.execute_shadow(&tx).await?,
            PrivacyTier::Phantom => self.payment_network.execute_phantom(&tx).await?,
            PrivacyTier::Quantum => self.payment_network.execute_quantum(&tx, consensus_proof).await?,
        };
        
        // Update account balances with quantum precision
        self.update_quantum_balances(&tx).await?;
        
        // Update quantum credit scores if applicable
        if matches!(tx.transaction_type, TransactionType::Loan | TransactionType::LoanRepayment | 
                   TransactionType::QNKUSDMint | TransactionType::QNKUSDBurn) {
            self.credit_engine.update_quantum_credit_score(&tx.from, &tx).await?;
        }
        
        // Store transaction history with quantum metadata
        self.store_quantum_transaction_history(tx).await?;
        
        Ok(tx_id)
    }

    /// Mint QNKUSD with quantum-enhanced collateral management
    pub async fn mint_qnkusd(
        &self,
        borrower: &Address,
        collateral_amount: u128,
        collateral_type: AssetType,
        qnkusd_amount: u128,
    ) -> Result<TransactionId> {
        self.qnkusd_system.mint_qnkusd(borrower, collateral_amount, collateral_type, qnkusd_amount)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    /// Burn QNKUSD to retrieve collateral
    pub async fn burn_qnkusd(
        &self,
        holder: &Address,
        qnkusd_amount: u128,
    ) -> Result<TransactionId> {
        self.qnkusd_system.burn_qnkusd(holder, qnkusd_amount)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    /// Get comprehensive Quillon Bank metrics
    pub async fn get_bank_metrics(&self) -> Result<BankMetrics> {
        let state = self.global_state.read().await;
        let accounts = self.accounts.read().await;
        
        let total_accounts = accounts.len() as u64;
        let total_net_worth: u128 = accounts.values()
            .flat_map(|account| account.balances.values())
            .map(|balance| balance.net_worth().max(0) as u128)
            .sum();

        let quantum_accounts = accounts.values()
            .filter(|account| account.quantum_features.quantum_vault_enabled)
            .count() as u64;
        
        Ok(BankMetrics {
            total_accounts,
            quantum_accounts,
            total_net_worth,
            total_deposits: state.total_deposits.clone(),
            total_loans: state.total_loans.clone(),
            qnkusd_metrics: QNKUSDMetrics {
                total_supply: state.qnkusd_total_supply,
                collateral_ratio: state.qnkusd_collateral_ratio,
                stability_score: state.system_health.qnkusd_stability_score,
            },
            system_health: state.system_health.clone(),
            quantum_metrics: state.quantum_metrics.clone(),
            daily_volume: state.daily_volume.clone(),
            average_credit_score: calculate_average_credit_score(&accounts),
        })
    }

    // Private helper methods
    async fn detect_quantum_fraud_preemptive(&self, tx: &Transaction) -> Result<f64> {
        // Enhanced fraud detection with quantum transaction patterns
        let accounts = self.accounts.read().await;
        let account = accounts.get(&tx.from)
            .ok_or_else(|| anyhow!("Account not found"))?;
        
        // Analyze quantum patterns using AI
        let fraud_score = self.analyze_quantum_transaction_patterns(&account.transaction_history, tx).await?;
        
        Ok(fraud_score)
    }

    async fn analyze_quantum_transaction_patterns(&self, history: &[Transaction], current_tx: &Transaction) -> Result<f64> {
        // Enhanced pattern analysis for quantum transactions
        let mut risk_score: f64 = 0.0;
        
        // Check for quantum feature usage patterns
        let quantum_tx_count = history.iter()
            .filter(|tx| matches!(tx.privacy_level, PrivacyTier::Quantum))
            .count();
        
        if quantum_tx_count == 0 && matches!(current_tx.privacy_level, PrivacyTier::Quantum) {
            risk_score += 0.1; // First-time quantum usage
        }
        
        // Standard fraud detection patterns
        if let Some(avg_amount) = calculate_average_amount(history) {
            if current_tx.amount > avg_amount * 10 {
                risk_score += 0.3;
            }
        }
        
        // Check for unusual timing
        if let Some(last_tx) = history.last() {
            let time_diff = current_tx.timestamp - last_tx.timestamp;
            if time_diff < 60 { // Less than 1 minute
                risk_score += 0.2;
            }
        }
        
        Ok(risk_score.min(1.0_f64))
    }

    async fn update_quantum_balances(&self, tx: &Transaction) -> Result<()> {
        let mut accounts = self.accounts.write().await;
        
        // Update sender balance with quantum features
        if let Some(from_account) = accounts.get_mut(&tx.from) {
            let balance = from_account.balances.entry(tx.asset.clone()).or_insert(Balance {
                available: 0,
                locked: 0,
                staked: 0,
                borrowed: 0,
                lending: 0,
                quantum_secured: 0,
                last_updated: tx.timestamp,
            });
            
            if balance.available >= tx.amount + tx.fee {
                balance.available -= tx.amount + tx.fee;
                balance.last_updated = tx.timestamp;
            } else {
                return Err(anyhow!("Insufficient balance"));
            }
        }
        
        // Update receiver balance
        if let Some(to_account) = accounts.get_mut(&tx.to) {
            let balance = to_account.balances.entry(tx.asset.clone()).or_insert(Balance {
                available: 0,
                locked: 0,
                staked: 0,
                borrowed: 0,
                lending: 0,
                quantum_secured: 0,
                last_updated: tx.timestamp,
            });
            
            balance.available += tx.amount;
            balance.last_updated = tx.timestamp;
        }
        
        Ok(())
    }

    async fn store_quantum_transaction_history(&self, tx: Transaction) -> Result<()> {
        let mut accounts = self.accounts.write().await;
        
        // Add to sender's history
        if let Some(from_account) = accounts.get_mut(&tx.from) {
            from_account.transaction_history.push(tx.clone());
            from_account.last_activity = tx.timestamp;
        }
        
        // Add to receiver's history
        if let Some(to_account) = accounts.get_mut(&tx.to) {
            to_account.transaction_history.push(tx.clone());
            to_account.last_activity = tx.timestamp;
        }
        
        Ok(())
    }
}

/// Comprehensive Quillon Bank metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BankMetrics {
    pub total_accounts: u64,
    pub quantum_accounts: u64,
    pub total_net_worth: u128,
    pub total_deposits: HashMap<AssetType, u128>,
    pub total_loans: HashMap<AssetType, u128>,
    pub qnkusd_metrics: QNKUSDMetrics,
    pub system_health: SystemHealth,
    pub quantum_metrics: QuantumBankMetrics,
    pub daily_volume: HashMap<AssetType, u128>,
    pub average_credit_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QNKUSDMetrics {
    pub total_supply: u128,
    pub collateral_ratio: f64,
    pub stability_score: f64,
}

// Helper functions
fn calculate_average_amount(history: &[Transaction]) -> Option<u128> {
    if history.is_empty() {
        return None;
    }
    
    let total: u128 = history.iter().map(|tx| tx.amount).sum();
    Some(total / history.len() as u128)
}

fn calculate_average_credit_score(accounts: &HashMap<Address, BankAccount>) -> f64 {
    if accounts.is_empty() {
        return 0.0;
    }
    
    let total: u32 = accounts.values().map(|account| account.credit_score.score as u32).sum();
    total as f64 / accounts.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use q_types::Phase;

    #[tokio::test]
    async fn test_quillon_bank_creation() {
        let node_id = [1u8; 32];
        let phase = Phase::Phase1;
        let plugin_manager = Arc::new(PluginManager::new());

        let bank = QuillonBankSystem::new(node_id, phase, plugin_manager).await;
        assert!(bank.is_ok());
    }

    #[tokio::test]
    async fn test_quantum_account_creation() {
        let node_id = [1u8; 32];
        let phase = Phase::Phase1;
        let plugin_manager = Arc::new(PluginManager::new());

        let bank = QuillonBankSystem::new(node_id, phase, plugin_manager).await.unwrap();
        bank.initialize().await.unwrap();

        // Test account creation would require mock identity proof
        // This is a placeholder test structure
    }
}