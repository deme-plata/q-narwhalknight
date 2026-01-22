//! BioDSL <-> Smart Contract Integration
//!
//! This module bridges the BioDSL synthesis engine with Q-NarwhalKnight
//! smart contracts for:
//!
//! - **License Verification**: Check on-chain licenses before synthesis
//! - **Proof Recording**: Record synthesis proofs on-chain
//! - **Token Economics**: Consume/earn BIO tokens for synthesis
//! - **Safety Oracle**: Query decentralized safety classifications
//! - **Marketplace**: List and order synthesis services
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                      BioDSL Synthesis Engine                     │
//! │                                                                  │
//! │  ┌────────────┐   ┌────────────┐   ┌────────────┐               │
//! │  │  Compiler  │──▶│  Executor  │──▶│  Verifier  │               │
//! │  └────────────┘   └────────────┘   └────────────┘               │
//! │         │               │               │                        │
//! └─────────┼───────────────┼───────────────┼────────────────────────┘
//!           │               │               │
//!           ▼               ▼               ▼
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                    BlockchainBridge (this module)                │
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
//! │  │ License  │  │ Synthesis│  │   Bio    │  │ Safety   │        │
//! │  │  Check   │  │  Proof   │  │  Token   │  │  Oracle  │        │
//! │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
//! └──────────────────────────────────────────────────────────────────┘
//!           │               │               │               │
//!           ▼               ▼               ▼               ▼
//! ┌──────────────────────────────────────────────────────────────────┐
//! │              Q-NarwhalKnight DAG-Knight Consensus                │
//! │         (Smart Contract State + 150K+ TPS Execution)             │
//! └──────────────────────────────────────────────────────────────────┘
//! ```

use crate::{BioDSLError, CompiledProgram, SynthesisResult};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;

/// DEA Schedule (mirrors smart contract)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DEASchedule {
    ScheduleI,
    ScheduleII,
    ScheduleIII,
    ScheduleIV,
    ScheduleV,
    ResearchOnly,
    Unscheduled,
}

/// Blockchain integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockchainConfig {
    /// RPC endpoint for Q-NarwhalKnight node
    pub rpc_endpoint: String,
    /// Wallet address for signing transactions
    pub wallet_address: u64,
    /// License contract address
    pub license_contract: [u8; 32],
    /// Synthesis proof contract address
    pub proof_contract: [u8; 32],
    /// BioToken contract address
    pub token_contract: [u8; 32],
    /// Safety oracle contract address
    pub oracle_contract: [u8; 32],
    /// Marketplace contract address
    pub marketplace_contract: [u8; 32],
    /// Gas limit for transactions
    pub gas_limit: u64,
}

impl Default for BlockchainConfig {
    fn default() -> Self {
        Self {
            rpc_endpoint: "http://localhost:8080".to_string(),
            wallet_address: 0,
            license_contract: [0u8; 32],
            proof_contract: [0u8; 32],
            token_contract: [0u8; 32],
            oracle_contract: [0u8; 32],
            marketplace_contract: [0u8; 32],
            gas_limit: 1_000_000,
        }
    }
}

/// On-chain license information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnChainLicense {
    pub license_id: [u8; 32],
    pub holder: u64,
    pub max_schedule: DEASchedule,
    pub authorized_molecules: Vec<[u8; 32]>,
    pub daily_quota_mg: HashMap<[u8; 32], u64>,
    pub expires_at: u64,
    pub active: bool,
}

/// Synthesis proof record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnChainProof {
    pub proof_id: [u8; 32],
    pub synthesizer: u64,
    pub molecule_hash: [u8; 32],
    pub molecule_name: String,
    pub quantity_ug: u64,
    pub purity_bps: u16,
    pub program_hash: [u8; 32],
    pub license_id: Option<[u8; 32]>,
    pub block_height: u64,
    pub tx_hash: [u8; 32],
}

/// Safety classification from oracle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyClassification {
    Unrestricted,
    Controlled(DEASchedule),
    Prohibited,
    Unknown,
}

/// Blockchain bridge for BioDSL integration
pub struct BlockchainBridge {
    config: BlockchainConfig,
    /// Cached license data
    license_cache: HashMap<u64, Vec<OnChainLicense>>,
    /// Cached safety classifications
    safety_cache: HashMap<[u8; 32], SafetyClassification>,
    /// Pending proofs to submit
    pending_proofs: Vec<PendingProof>,
    /// Token balance cache
    token_balance: u128,
    /// Synthesis credits available
    synthesis_credits: u64,
}

#[derive(Debug, Clone)]
struct PendingProof {
    molecule_name: String,
    smiles: Option<String>,
    quantity_ug: u64,
    purity_bps: u16,
    program_hash: [u8; 32],
    license_id: Option<[u8; 32]>,
    swarm_config_hash: [u8; 32],
    started_at: u64,
}

impl BlockchainBridge {
    /// Create a new blockchain bridge
    pub fn new(config: BlockchainConfig) -> Self {
        Self {
            config,
            license_cache: HashMap::new(),
            safety_cache: HashMap::new(),
            pending_proofs: Vec::new(),
            token_balance: 0,
            synthesis_credits: 0,
        }
    }

    /// Check if user has valid license for molecule synthesis
    pub async fn check_license(
        &mut self,
        molecule_name: &str,
        smiles: Option<&str>,
        quantity_mg: u64,
        required_schedule: DEASchedule,
    ) -> Result<LicenseCheckResult, BioDSLError> {
        let molecule_hash = Self::hash_molecule(molecule_name, smiles);
        let holder = self.config.wallet_address;

        // Check cache first
        if let Some(licenses) = self.license_cache.get(&holder) {
            for license in licenses {
                if !license.active {
                    continue;
                }

                // Check expiration
                let now = current_timestamp();
                if license.expires_at < now {
                    continue;
                }

                // Check schedule authorization
                if !Self::schedule_permits(&license.max_schedule, &required_schedule) {
                    continue;
                }

                // Check molecule authorization
                if !license.authorized_molecules.is_empty()
                    && !license.authorized_molecules.contains(&molecule_hash)
                {
                    continue;
                }

                // Check quota
                if let Some(&quota) = license.daily_quota_mg.get(&molecule_hash) {
                    if quantity_mg > quota {
                        return Ok(LicenseCheckResult::QuotaExceeded {
                            requested: quantity_mg,
                            available: quota,
                        });
                    }
                }

                return Ok(LicenseCheckResult::Valid {
                    license_id: license.license_id,
                });
            }
        }

        // No valid license found
        if required_schedule == DEASchedule::Unscheduled {
            Ok(LicenseCheckResult::NotRequired)
        } else {
            Ok(LicenseCheckResult::Required {
                schedule: required_schedule,
            })
        }
    }

    /// Query safety oracle for molecule classification
    pub async fn query_safety(
        &mut self,
        molecule_name: &str,
        smiles: Option<&str>,
    ) -> Result<SafetyClassification, BioDSLError> {
        let molecule_hash = Self::hash_molecule(molecule_name, smiles);

        // Check cache
        if let Some(&classification) = self.safety_cache.get(&molecule_hash) {
            return Ok(classification);
        }

        // In production, this would query the on-chain oracle
        // For now, use local biosafety rules
        let classification = self.local_safety_check(molecule_name);
        self.safety_cache.insert(molecule_hash, classification);

        Ok(classification)
    }

    /// Check token balance and synthesis credits
    pub async fn check_credits(&mut self) -> Result<CreditStatus, BioDSLError> {
        // In production, query BioToken contract
        Ok(CreditStatus {
            token_balance: self.token_balance,
            synthesis_credits: self.synthesis_credits,
            can_synthesize: self.synthesis_credits > 0,
        })
    }

    /// Consume a synthesis credit before synthesis
    pub async fn consume_credit(&mut self) -> Result<(), BioDSLError> {
        if self.synthesis_credits == 0 {
            return Err(BioDSLError::ExecutionError(
                "No synthesis credits available. Stake BIO tokens to earn credits.".to_string(),
            ));
        }

        if self.synthesis_credits != u64::MAX {
            self.synthesis_credits -= 1;
        }

        Ok(())
    }

    /// Pre-synthesis validation (license + safety + credits)
    pub async fn pre_synthesis_check(
        &mut self,
        program: &CompiledProgram,
        molecule_name: &str,
        smiles: Option<&str>,
        quantity_mg: u64,
    ) -> Result<PreSynthesisApproval, BioDSLError> {
        // 1. Query safety oracle
        let safety = self.query_safety(molecule_name, smiles).await?;

        let required_schedule = match safety {
            SafetyClassification::Prohibited => {
                return Err(BioDSLError::SafetyViolation(format!(
                    "Molecule '{}' is prohibited for synthesis",
                    molecule_name
                )));
            }
            SafetyClassification::Controlled(sched) => sched,
            SafetyClassification::Unknown => {
                return Err(BioDSLError::SafetyViolation(format!(
                    "Molecule '{}' has unknown safety classification. Oracle vote required.",
                    molecule_name
                )));
            }
            SafetyClassification::Unrestricted => DEASchedule::Unscheduled,
        };

        // 2. Check license
        let license_result = self.check_license(
            molecule_name,
            smiles,
            quantity_mg,
            required_schedule,
        ).await?;

        let license_id = match license_result {
            LicenseCheckResult::Valid { license_id } => Some(license_id),
            LicenseCheckResult::NotRequired => None,
            LicenseCheckResult::Required { schedule } => {
                return Err(BioDSLError::SafetyViolation(format!(
                    "License required for Schedule {:?} substance",
                    schedule
                )));
            }
            LicenseCheckResult::QuotaExceeded { requested, available } => {
                return Err(BioDSLError::SafetyViolation(format!(
                    "Daily quota exceeded: requested {}mg, only {}mg available",
                    requested, available
                )));
            }
        };

        // 3. Check synthesis credits
        let credits = self.check_credits().await?;
        if !credits.can_synthesize {
            return Err(BioDSLError::ExecutionError(
                "No synthesis credits. Stake BIO tokens first.".to_string(),
            ));
        }

        // 4. Prepare pending proof
        let program_hash = Self::hash_program(program);
        let pending = PendingProof {
            molecule_name: molecule_name.to_string(),
            smiles: smiles.map(|s| s.to_string()),
            quantity_ug: quantity_mg * 1000,
            purity_bps: 9900, // 99% default
            program_hash,
            license_id,
            swarm_config_hash: [0u8; 32], // Would be set by swarm
            started_at: current_timestamp(),
        };
        self.pending_proofs.push(pending);

        Ok(PreSynthesisApproval {
            safety_classification: safety,
            license_id,
            credits_remaining: credits.synthesis_credits,
            program_hash,
        })
    }

    /// Record synthesis proof on-chain after successful synthesis
    pub async fn record_synthesis_proof(
        &mut self,
        result: &SynthesisResult,
        purity_bps: u16,
        tomography_signature: [u8; 64],
    ) -> Result<OnChainProof, BioDSLError> {
        let pending = self.pending_proofs.pop()
            .ok_or_else(|| BioDSLError::ExecutionError(
                "No pending synthesis to record".to_string()
            ))?;

        // In production, this would submit a transaction to the proof contract
        let proof_id = Self::compute_proof_id(
            self.config.wallet_address,
            &pending.molecule_name,
            pending.smiles.as_deref(),
            current_timestamp(),
            pending.program_hash,
        );

        let proof = OnChainProof {
            proof_id,
            synthesizer: self.config.wallet_address,
            molecule_hash: Self::hash_molecule(&pending.molecule_name, pending.smiles.as_deref()),
            molecule_name: pending.molecule_name,
            quantity_ug: pending.quantity_ug,
            purity_bps,
            program_hash: pending.program_hash,
            license_id: pending.license_id,
            block_height: 0, // Would be set by chain
            tx_hash: [0u8; 32], // Would be set by tx
        };

        // Consume credit
        self.consume_credit().await?;

        // In production, earn synthesis reward
        self.token_balance += 1_000_000_000_000_000_000_000_000; // 1 BIO

        Ok(proof)
    }

    /// List synthesis service on marketplace
    pub async fn list_service(
        &mut self,
        molecule_name: &str,
        smiles: Option<&str>,
        price_per_mg: u128,
        min_quantity_mg: u64,
        max_quantity_mg: u64,
        estimated_time_hours: u32,
        purity_guarantee_bps: u16,
    ) -> Result<[u8; 32], BioDSLError> {
        let molecule_hash = Self::hash_molecule(molecule_name, smiles);

        // In production, this would create a marketplace listing transaction
        let listing_id = {
            let mut hasher = Sha3_256::new();
            hasher.update(self.config.wallet_address.to_le_bytes());
            hasher.update(molecule_hash);
            hasher.update(current_timestamp().to_le_bytes());
            hasher.update(b"LISTING_V1");
            hasher.finalize().into()
        };

        Ok(listing_id)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Helper Functions
    // ═══════════════════════════════════════════════════════════════════════════

    fn hash_molecule(name: &str, smiles: Option<&str>) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(name.to_lowercase().as_bytes());
        if let Some(s) = smiles {
            hasher.update(s.as_bytes());
        }
        hasher.finalize().into()
    }

    fn hash_program(program: &CompiledProgram) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(format!("{:?}", program.instructions).as_bytes());
        hasher.finalize().into()
    }

    fn compute_proof_id(
        synthesizer: u64,
        molecule_name: &str,
        smiles: Option<&str>,
        timestamp: u64,
        program_hash: [u8; 32],
    ) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(synthesizer.to_le_bytes());
        hasher.update(Self::hash_molecule(molecule_name, smiles));
        hasher.update(timestamp.to_le_bytes());
        hasher.update(program_hash);
        hasher.update(b"SYNTHESIS_PROOF_V1");
        hasher.finalize().into()
    }

    fn schedule_permits(max: &DEASchedule, required: &DEASchedule) -> bool {
        use DEASchedule::*;
        match (max, required) {
            (ScheduleI, _) => true,
            (ScheduleII, ScheduleII | ScheduleIII | ScheduleIV | ScheduleV | ResearchOnly | Unscheduled) => true,
            (ScheduleIII, ScheduleIII | ScheduleIV | ScheduleV | Unscheduled) => true,
            (ScheduleIV, ScheduleIV | ScheduleV | Unscheduled) => true,
            (ScheduleV, ScheduleV | Unscheduled) => true,
            (ResearchOnly, ResearchOnly | Unscheduled) => true,
            (Unscheduled, Unscheduled) => true,
            _ => false,
        }
    }

    fn local_safety_check(&self, molecule_name: &str) -> SafetyClassification {
        // Local fallback safety rules
        match molecule_name.to_lowercase().as_str() {
            // Prohibited
            "ricin" | "anthrax" | "botulinum" | "vx" | "sarin" | "novichok" => {
                SafetyClassification::Prohibited
            }
            // Schedule I
            "thc" | "psilocybin" | "dmt" | "lsd" | "mescaline" | "mdma" => {
                SafetyClassification::Controlled(DEASchedule::ScheduleI)
            }
            // Schedule II
            "morphine" | "fentanyl" | "amphetamine" | "cocaine" => {
                SafetyClassification::Controlled(DEASchedule::ScheduleII)
            }
            // Schedule III
            "ketamine" | "codeine" => {
                SafetyClassification::Controlled(DEASchedule::ScheduleIII)
            }
            // Unrestricted
            "caffeine" | "cbd" | "aspirin" | "ibuprofen" | "melatonin" |
            "dopamine" | "limonene" | "myrcene" | "pinene" | "water" => {
                SafetyClassification::Unrestricted
            }
            // Unknown - requires oracle vote
            _ => SafetyClassification::Unknown
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of license check
#[derive(Debug, Clone)]
pub enum LicenseCheckResult {
    /// Valid license found
    Valid { license_id: [u8; 32] },
    /// License not required for this molecule
    NotRequired,
    /// License required but not found
    Required { schedule: DEASchedule },
    /// Daily quota exceeded
    QuotaExceeded { requested: u64, available: u64 },
}

/// Credit status
#[derive(Debug, Clone)]
pub struct CreditStatus {
    pub token_balance: u128,
    pub synthesis_credits: u64,
    pub can_synthesize: bool,
}

/// Pre-synthesis approval
#[derive(Debug, Clone)]
pub struct PreSynthesisApproval {
    pub safety_classification: SafetyClassification,
    pub license_id: Option<[u8; 32]>,
    pub credits_remaining: u64,
    pub program_hash: [u8; 32],
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper
// ═══════════════════════════════════════════════════════════════════════════════

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_safety_check() {
        let mut bridge = BlockchainBridge::new(BlockchainConfig::default());

        // Prohibited substance
        let result = bridge.query_safety("ricin", None).await.unwrap();
        assert_eq!(result, SafetyClassification::Prohibited);

        // Schedule I
        let result = bridge.query_safety("THC", None).await.unwrap();
        assert!(matches!(result, SafetyClassification::Controlled(DEASchedule::ScheduleI)));

        // Unrestricted
        let result = bridge.query_safety("caffeine", None).await.unwrap();
        assert_eq!(result, SafetyClassification::Unrestricted);
    }

    #[tokio::test]
    async fn test_license_not_required() {
        let mut bridge = BlockchainBridge::new(BlockchainConfig::default());

        let result = bridge.check_license(
            "caffeine",
            None,
            1000,
            DEASchedule::Unscheduled,
        ).await.unwrap();

        assert!(matches!(result, LicenseCheckResult::NotRequired));
    }

    #[test]
    fn test_molecule_hashing() {
        let hash1 = BlockchainBridge::hash_molecule("THC", None);
        let hash2 = BlockchainBridge::hash_molecule("thc", None);
        assert_eq!(hash1, hash2); // Case insensitive
    }
}
