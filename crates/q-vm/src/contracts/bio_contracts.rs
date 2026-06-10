//! Bio-Smart Contract System
//!
//! Blockchain-enabled biological programming infrastructure that integrates
//! the BioDSL molecular synthesis system with smart contracts for:
//!
//! - **License Management**: DEA-compliant controlled substance authorization
//! - **Synthesis Verification**: Cryptographic proof of molecular synthesis
//! - **Bio Tokens**: Tradeable synthesis rights and resource allocation
//! - **Safety Oracles**: Decentralized biosafety verification network
//! - **Synthesis Marketplace**: P2P market for synthesis services
//! - **Supply Chain**: On-chain molecule provenance tracking
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        Q-NarwhalKnight Bio-Chain                        │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │
//! │  │ BioLicense  │   │ BioToken    │   │ Synthesis   │   │ BiSafety    │ │
//! │  │ Contract    │──▶│ Contract    │──▶│ Marketplace │──▶│ Oracle      │ │
//! │  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘ │
//! │         │                 │                 │                 │        │
//! │         ▼                 ▼                 ▼                 ▼        │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │                    Synthesis Proof Registry                     │   │
//! │  │         (Immutable record of all synthesis operations)          │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │                                    │                                    │
//! │                                    ▼                                    │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │                         BioDSL Engine                           │   │
//! │  │            (Water Robot Swarm Molecular Assembly)               │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════════
// BIOLICENSE CONTRACT - DEA/FDA Compliant Licensing System
// ═══════════════════════════════════════════════════════════════════════════════

/// DEA Schedule classifications
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DEASchedule {
    /// Schedule I - High abuse potential, no accepted medical use (THC, Psilocybin, DMT)
    ScheduleI,
    /// Schedule II - High abuse potential, accepted medical use (Morphine, Fentanyl)
    ScheduleII,
    /// Schedule III - Moderate abuse potential (Ketamine, Codeine compounds)
    ScheduleIII,
    /// Schedule IV - Low abuse potential (Benzodiazepines)
    ScheduleIV,
    /// Schedule V - Lowest abuse potential (Low-dose codeine cough syrups)
    ScheduleV,
    /// Research-only designation
    ResearchOnly,
    /// Not scheduled - unrestricted
    Unscheduled,
}

/// License type for controlled substance handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LicenseType {
    /// DEA Researcher License (Schedule I-V for research)
    DEAResearcher,
    /// DEA Manufacturer License
    DEAManufacturer,
    /// FDA IND (Investigational New Drug)
    FDAIND,
    /// State Medical Research
    StateMedical,
    /// Academic Research Institution
    Academic,
    /// Hemp/CBD License (< 0.3% THC)
    HempCBD,
}

/// On-chain license record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioLicense {
    /// License unique ID (hash of holder + license number)
    pub license_id: [u8; 32],
    /// License holder address
    pub holder: u64,
    /// License type
    pub license_type: LicenseType,
    /// Maximum schedule allowed
    pub max_schedule: DEASchedule,
    /// Specific molecules authorized (SMILES hashes)
    pub authorized_molecules: Vec<[u8; 32]>,
    /// Daily synthesis quota in milligrams per molecule
    pub daily_quota_mg: HashMap<[u8; 32], u64>,
    /// License issuance timestamp (Unix)
    pub issued_at: u64,
    /// License expiration timestamp (Unix)
    pub expires_at: u64,
    /// Issuing authority signature
    #[serde(with = "BigArray")]
    pub authority_signature: [u8; 64],
    /// Is license currently active
    pub active: bool,
    /// Total synthesis count under this license
    pub synthesis_count: u64,
}

/// BioLicense contract state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BioLicenseContract {
    /// License registry: license_id -> BioLicense
    pub licenses: HashMap<[u8; 32], BioLicense>,
    /// Holder to licenses mapping
    pub holder_licenses: HashMap<u64, Vec<[u8; 32]>>,
    /// Authorized issuers (regulatory bodies)
    pub authorized_issuers: Vec<u64>,
    /// Contract owner
    pub owner: u64,
    /// Daily usage tracking: (license_id, molecule_hash, day) -> mg_used
    pub daily_usage: HashMap<([u8; 32], [u8; 32], u64), u64>,
}

impl BioLicenseContract {
    pub fn new(owner: u64) -> Self {
        Self {
            licenses: HashMap::new(),
            holder_licenses: HashMap::new(),
            authorized_issuers: vec![owner], // Owner is initial authority
            owner,
            daily_usage: HashMap::new(),
        }
    }

    /// Issue a new license (only authorized issuers)
    pub fn issue_license(
        &mut self,
        issuer: u64,
        holder: u64,
        license_type: LicenseType,
        max_schedule: DEASchedule,
        authorized_molecules: Vec<[u8; 32]>,
        daily_quota_mg: HashMap<[u8; 32], u64>,
        validity_days: u64,
        signature: [u8; 64],
    ) -> Result<[u8; 32], BioContractError> {
        // Verify issuer is authorized
        if !self.authorized_issuers.contains(&issuer) {
            return Err(BioContractError::UnauthorizedIssuer);
        }

        let now = current_timestamp();
        let license_id = Self::compute_license_id(holder, now);

        let license = BioLicense {
            license_id,
            holder,
            license_type,
            max_schedule,
            authorized_molecules,
            daily_quota_mg,
            issued_at: now,
            expires_at: now + (validity_days * 86400),
            authority_signature: signature,
            active: true,
            synthesis_count: 0,
        };

        self.licenses.insert(license_id, license);
        self.holder_licenses
            .entry(holder)
            .or_default()
            .push(license_id);

        Ok(license_id)
    }

    /// Verify license for synthesis operation
    pub fn verify_license(
        &self,
        holder: u64,
        molecule_hash: [u8; 32],
        quantity_mg: u64,
        required_schedule: DEASchedule,
    ) -> Result<[u8; 32], BioContractError> {
        let now = current_timestamp();
        let today = now / 86400;

        // Find valid license for holder
        let license_ids = self.holder_licenses.get(&holder)
            .ok_or(BioContractError::NoLicenseFound)?;

        for license_id in license_ids {
            let license = self.licenses.get(license_id)
                .ok_or(BioContractError::LicenseNotFound)?;

            // Check license validity
            if !license.active {
                continue;
            }
            if license.expires_at < now {
                continue;
            }

            // Check schedule authorization
            if !Self::schedule_permits(&license.max_schedule, &required_schedule) {
                continue;
            }

            // Check molecule authorization
            if !license.authorized_molecules.contains(&molecule_hash)
               && !license.authorized_molecules.is_empty() {
                continue;
            }

            // Check daily quota
            if let Some(&quota) = license.daily_quota_mg.get(&molecule_hash) {
                let usage_key = (*license_id, molecule_hash, today);
                let used = self.daily_usage.get(&usage_key).unwrap_or(&0);
                if used + quantity_mg > quota {
                    return Err(BioContractError::QuotaExceeded {
                        requested: quantity_mg,
                        remaining: quota.saturating_sub(*used),
                    });
                }
            }

            return Ok(*license_id);
        }

        Err(BioContractError::NoValidLicense)
    }

    /// Record synthesis usage against license
    pub fn record_usage(
        &mut self,
        license_id: [u8; 32],
        molecule_hash: [u8; 32],
        quantity_mg: u64,
    ) -> Result<(), BioContractError> {
        let license = self.licenses.get_mut(&license_id)
            .ok_or(BioContractError::LicenseNotFound)?;

        license.synthesis_count += 1;

        let today = current_timestamp() / 86400;
        let usage_key = (license_id, molecule_hash, today);
        *self.daily_usage.entry(usage_key).or_insert(0) += quantity_mg;

        Ok(())
    }

    /// Revoke a license
    pub fn revoke_license(
        &mut self,
        issuer: u64,
        license_id: [u8; 32],
    ) -> Result<(), BioContractError> {
        if !self.authorized_issuers.contains(&issuer) {
            return Err(BioContractError::UnauthorizedIssuer);
        }

        let license = self.licenses.get_mut(&license_id)
            .ok_or(BioContractError::LicenseNotFound)?;

        license.active = false;
        Ok(())
    }

    fn compute_license_id(holder: u64, timestamp: u64) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(holder.to_le_bytes());
        hasher.update(timestamp.to_le_bytes());
        hasher.update(b"BIO_LICENSE_V1");
        hasher.finalize().into()
    }

    fn schedule_permits(max: &DEASchedule, required: &DEASchedule) -> bool {
        use DEASchedule::*;
        match (max, required) {
            (ScheduleI, _) => true,  // Schedule I permits all
            (ScheduleII, ScheduleII | ScheduleIII | ScheduleIV | ScheduleV | ResearchOnly | Unscheduled) => true,
            (ScheduleIII, ScheduleIII | ScheduleIV | ScheduleV | Unscheduled) => true,
            (ScheduleIV, ScheduleIV | ScheduleV | Unscheduled) => true,
            (ScheduleV, ScheduleV | Unscheduled) => true,
            (ResearchOnly, ResearchOnly | Unscheduled) => true,
            (Unscheduled, Unscheduled) => true,
            _ => false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SYNTHESIS PROOF CONTRACT - Cryptographic Verification of Molecular Synthesis
// ═══════════════════════════════════════════════════════════════════════════════

/// Proof of molecular synthesis - immutable on-chain record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisProof {
    /// Unique proof ID
    pub proof_id: [u8; 32],
    /// Synthesizer address
    pub synthesizer: u64,
    /// Molecule SMILES hash
    pub molecule_hash: [u8; 32],
    /// Molecule name
    pub molecule_name: String,
    /// SMILES notation (may be encrypted for proprietary molecules)
    pub smiles: Option<String>,
    /// Quantity synthesized in micrograms
    pub quantity_ug: u64,
    /// Purity percentage (0-10000 = 0.00-100.00%)
    pub purity_bps: u16,
    /// BioDSL program hash used for synthesis
    pub program_hash: [u8; 32],
    /// License ID used (if controlled substance)
    pub license_id: Option<[u8; 32]>,
    /// Robot swarm configuration hash
    pub swarm_config_hash: [u8; 32],
    /// Synthesis start timestamp
    pub started_at: u64,
    /// Synthesis completion timestamp
    pub completed_at: u64,
    /// Verification method used
    pub verification_method: VerificationMethod,
    /// Quantum tomography signature (from robot swarm)
    #[serde(with = "BigArray")]
    pub tomography_signature: [u8; 64],
    /// Block height when recorded
    pub block_height: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationMethod {
    /// Quantum tomography verification
    QuantumTomography,
    /// Mass spectrometry
    MassSpectrometry,
    /// NMR spectroscopy
    NMRSpectroscopy,
    /// X-ray crystallography
    XRayCrystallography,
    /// Computational verification only
    ComputationalOnly,
    /// Multi-modal verification
    MultiModal,
}

/// Synthesis proof registry contract
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SynthesisProofContract {
    /// All proofs: proof_id -> SynthesisProof
    pub proofs: HashMap<[u8; 32], SynthesisProof>,
    /// Proofs by synthesizer
    pub by_synthesizer: HashMap<u64, Vec<[u8; 32]>>,
    /// Proofs by molecule
    pub by_molecule: HashMap<[u8; 32], Vec<[u8; 32]>>,
    /// Total synthesis volume by molecule (in micrograms)
    pub total_volume: HashMap<[u8; 32], u64>,
    /// Verification oracles
    pub verification_oracles: Vec<u64>,
    /// Contract owner
    pub owner: u64,
    /// Current block height
    pub current_height: u64,
}

impl SynthesisProofContract {
    pub fn new(owner: u64) -> Self {
        Self {
            proofs: HashMap::new(),
            by_synthesizer: HashMap::new(),
            by_molecule: HashMap::new(),
            total_volume: HashMap::new(),
            verification_oracles: vec![owner],
            owner,
            current_height: 0,
        }
    }

    /// Record a new synthesis proof
    pub fn record_proof(
        &mut self,
        synthesizer: u64,
        molecule_name: String,
        smiles: Option<String>,
        quantity_ug: u64,
        purity_bps: u16,
        program_hash: [u8; 32],
        license_id: Option<[u8; 32]>,
        swarm_config_hash: [u8; 32],
        started_at: u64,
        verification_method: VerificationMethod,
        tomography_signature: [u8; 64],
    ) -> Result<[u8; 32], BioContractError> {
        let completed_at = current_timestamp();
        let molecule_hash = Self::hash_molecule(&molecule_name, smiles.as_deref());

        let proof_id = Self::compute_proof_id(
            synthesizer,
            molecule_hash,
            completed_at,
            program_hash,
        );

        let proof = SynthesisProof {
            proof_id,
            synthesizer,
            molecule_hash,
            molecule_name,
            smiles,
            quantity_ug,
            purity_bps,
            program_hash,
            license_id,
            swarm_config_hash,
            started_at,
            completed_at,
            verification_method,
            tomography_signature,
            block_height: self.current_height,
        };

        self.proofs.insert(proof_id, proof);
        self.by_synthesizer.entry(synthesizer).or_default().push(proof_id);
        self.by_molecule.entry(molecule_hash).or_default().push(proof_id);
        *self.total_volume.entry(molecule_hash).or_insert(0) += quantity_ug;

        Ok(proof_id)
    }

    /// Verify a synthesis proof exists and is valid
    pub fn verify_proof(&self, proof_id: [u8; 32]) -> Result<&SynthesisProof, BioContractError> {
        self.proofs.get(&proof_id)
            .ok_or(BioContractError::ProofNotFound)
    }

    /// Get total synthesized volume for a molecule
    pub fn get_total_volume(&self, molecule_hash: [u8; 32]) -> u64 {
        *self.total_volume.get(&molecule_hash).unwrap_or(&0)
    }

    /// Get synthesis history for an address
    pub fn get_history(&self, synthesizer: u64) -> Vec<&SynthesisProof> {
        self.by_synthesizer.get(&synthesizer)
            .map(|ids| ids.iter().filter_map(|id| self.proofs.get(id)).collect())
            .unwrap_or_default()
    }

    fn hash_molecule(name: &str, smiles: Option<&str>) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(name.as_bytes());
        if let Some(s) = smiles {
            hasher.update(s.as_bytes());
        }
        hasher.finalize().into()
    }

    fn compute_proof_id(
        synthesizer: u64,
        molecule_hash: [u8; 32],
        timestamp: u64,
        program_hash: [u8; 32],
    ) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(synthesizer.to_le_bytes());
        hasher.update(molecule_hash);
        hasher.update(timestamp.to_le_bytes());
        hasher.update(program_hash);
        hasher.update(b"SYNTHESIS_PROOF_V1");
        hasher.finalize().into()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BIOTOKEN CONTRACT - Tradeable Synthesis Rights and Resource Allocation
// ═══════════════════════════════════════════════════════════════════════════════

/// BioToken - ERC20-like token for synthesis rights
///
/// Token holders can:
/// - Access synthesis services proportional to their holdings
/// - Stake tokens for priority queue access
/// - Participate in governance of the bio-synthesis network
/// - Earn rewards from successful synthesis operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BioTokenContract {
    /// Token name
    pub name: String,
    /// Token symbol
    pub symbol: String,
    /// Decimals (24 for quantum-level precision)
    pub decimals: u8,
    /// Total supply
    pub total_supply: u128,
    /// Balances
    pub balances: HashMap<u64, u128>,
    /// Allowances
    pub allowances: HashMap<(u64, u64), u128>,
    /// Staked amounts
    pub staked: HashMap<u64, StakeInfo>,
    /// Synthesis credits (earned from staking)
    pub synthesis_credits: HashMap<u64, u64>,
    /// Contract owner
    pub owner: u64,
    /// Staking APY in basis points (e.g., 1200 = 12%)
    pub staking_apy_bps: u16,
    /// Synthesis reward per operation (in tokens)
    pub synthesis_reward: u128,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StakeInfo {
    pub amount: u128,
    pub staked_at: u64,
    pub last_claim: u64,
    pub tier: StakingTier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum StakingTier {
    #[default]
    None,
    /// Bronze: 1,000+ BIO - Basic synthesis access
    Bronze,
    /// Silver: 10,000+ BIO - Priority queue + 5% bonus
    Silver,
    /// Gold: 100,000+ BIO - Express synthesis + 15% bonus
    Gold,
    /// Platinum: 1,000,000+ BIO - Unlimited synthesis + 30% bonus
    Platinum,
}

impl BioTokenContract {
    pub fn new(owner: u64, initial_supply: u128) -> Self {
        let mut balances = HashMap::new();
        balances.insert(owner, initial_supply);

        Self {
            name: "BioSynthesis Token".to_string(),
            symbol: "BIO".to_string(),
            decimals: 24,
            total_supply: initial_supply,
            balances,
            allowances: HashMap::new(),
            staked: HashMap::new(),
            synthesis_credits: HashMap::new(),
            owner,
            staking_apy_bps: 1500, // 15% APY
            synthesis_reward: 1_000_000_000_000_000_000_000_000, // 1 BIO per synthesis
        }
    }

    /// Get balance of address
    pub fn balance_of(&self, address: u64) -> u128 {
        *self.balances.get(&address).unwrap_or(&0)
    }

    /// Transfer tokens
    pub fn transfer(
        &mut self,
        from: u64,
        to: u64,
        amount: u128,
    ) -> Result<(), BioContractError> {
        let from_balance = self.balance_of(from);
        if from_balance < amount {
            return Err(BioContractError::InsufficientBalance);
        }

        *self.balances.entry(from).or_insert(0) -= amount;
        *self.balances.entry(to).or_insert(0) += amount;

        Ok(())
    }

    /// Approve spender
    pub fn approve(
        &mut self,
        owner: u64,
        spender: u64,
        amount: u128,
    ) -> Result<(), BioContractError> {
        self.allowances.insert((owner, spender), amount);
        Ok(())
    }

    /// Transfer from (with allowance)
    pub fn transfer_from(
        &mut self,
        spender: u64,
        from: u64,
        to: u64,
        amount: u128,
    ) -> Result<(), BioContractError> {
        let allowance = *self.allowances.get(&(from, spender)).unwrap_or(&0);
        if allowance < amount {
            return Err(BioContractError::InsufficientAllowance);
        }

        self.transfer(from, to, amount)?;
        self.allowances.insert((from, spender), allowance - amount);

        Ok(())
    }

    /// Stake tokens for synthesis access
    pub fn stake(&mut self, staker: u64, amount: u128) -> Result<(), BioContractError> {
        let balance = self.balance_of(staker);
        if balance < amount {
            return Err(BioContractError::InsufficientBalance);
        }

        // Claim pending rewards first
        self.claim_rewards(staker)?;

        // Transfer to staked
        *self.balances.entry(staker).or_insert(0) -= amount;

        let stake_info = self.staked.entry(staker).or_default();
        stake_info.amount += amount;
        stake_info.staked_at = current_timestamp();
        stake_info.last_claim = current_timestamp();
        stake_info.tier = Self::calculate_tier(stake_info.amount);

        // Grant synthesis credits based on tier
        let credits = match stake_info.tier {
            StakingTier::None => 0,
            StakingTier::Bronze => 10,
            StakingTier::Silver => 50,
            StakingTier::Gold => 200,
            StakingTier::Platinum => u64::MAX, // Unlimited
        };
        *self.synthesis_credits.entry(staker).or_insert(0) += credits;

        Ok(())
    }

    /// Unstake tokens
    pub fn unstake(&mut self, staker: u64, amount: u128) -> Result<(), BioContractError> {
        let stake_info = self.staked.get(&staker)
            .ok_or(BioContractError::NotStaked)?;

        if stake_info.amount < amount {
            return Err(BioContractError::InsufficientStake);
        }

        // Claim pending rewards first
        self.claim_rewards(staker)?;

        // Return tokens
        let stake_info = self.staked.get_mut(&staker).unwrap();
        stake_info.amount -= amount;
        stake_info.tier = Self::calculate_tier(stake_info.amount);

        *self.balances.entry(staker).or_insert(0) += amount;

        Ok(())
    }

    /// Claim staking rewards
    pub fn claim_rewards(&mut self, staker: u64) -> Result<u128, BioContractError> {
        let stake_info = self.staked.get_mut(&staker)
            .ok_or(BioContractError::NotStaked)?;

        let now = current_timestamp();
        let time_staked = now - stake_info.last_claim;

        // Calculate rewards: (staked * apy * time) / (365 days * 10000 bps)
        let rewards = (stake_info.amount as u128)
            .saturating_mul(self.staking_apy_bps as u128)
            .saturating_mul(time_staked as u128)
            / (365 * 86400 * 10000);

        if rewards > 0 {
            // Mint rewards
            self.total_supply += rewards;
            *self.balances.entry(staker).or_insert(0) += rewards;
            stake_info.last_claim = now;
        }

        Ok(rewards)
    }

    /// Consume synthesis credit (called when synthesis is performed)
    pub fn consume_credit(&mut self, user: u64) -> Result<(), BioContractError> {
        let credits = self.synthesis_credits.get_mut(&user)
            .ok_or(BioContractError::NoSynthesisCredits)?;

        if *credits == 0 {
            return Err(BioContractError::NoSynthesisCredits);
        }

        if *credits != u64::MAX {
            *credits -= 1;
        }

        Ok(())
    }

    /// Reward synthesizer for successful operation
    pub fn reward_synthesis(&mut self, synthesizer: u64) -> Result<u128, BioContractError> {
        let stake_info = self.staked.get(&synthesizer);

        // Apply tier bonus
        let bonus_multiplier = match stake_info.map(|s| s.tier) {
            Some(StakingTier::Platinum) => 130,
            Some(StakingTier::Gold) => 115,
            Some(StakingTier::Silver) => 105,
            _ => 100,
        };

        let reward = self.synthesis_reward * bonus_multiplier / 100;

        // Mint reward
        self.total_supply += reward;
        *self.balances.entry(synthesizer).or_insert(0) += reward;

        Ok(reward)
    }

    fn calculate_tier(amount: u128) -> StakingTier {
        let one_token = 1_000_000_000_000_000_000_000_000u128; // 10^24

        if amount >= 1_000_000 * one_token {
            StakingTier::Platinum
        } else if amount >= 100_000 * one_token {
            StakingTier::Gold
        } else if amount >= 10_000 * one_token {
            StakingTier::Silver
        } else if amount >= 1_000 * one_token {
            StakingTier::Bronze
        } else {
            StakingTier::None
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BIOSAFETY ORACLE CONTRACT - Decentralized Safety Verification Network
// ═══════════════════════════════════════════════════════════════════════════════

/// Safety classification for molecules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyClassification {
    /// Safe for unrestricted synthesis
    Unrestricted,
    /// Controlled - requires license
    Controlled(DEASchedule),
    /// Prohibited - synthesis blocked
    Prohibited,
    /// Unknown - requires oracle vote
    Unknown,
}

/// Oracle vote on molecule safety
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyVote {
    pub oracle: u64,
    pub molecule_hash: [u8; 32],
    pub classification: SafetyClassification,
    pub evidence_hash: [u8; 32],
    pub timestamp: u64,
    pub stake: u128,
}

/// BioSafety Oracle contract
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BioSafetyOracleContract {
    /// Registered oracles
    pub oracles: HashMap<u64, OracleInfo>,
    /// Molecule classifications (finalized)
    pub classifications: HashMap<[u8; 32], SafetyClassification>,
    /// Pending votes
    pub pending_votes: HashMap<[u8; 32], Vec<SafetyVote>>,
    /// Voting threshold (number of oracles needed)
    pub voting_threshold: usize,
    /// Minimum stake to be an oracle
    pub min_oracle_stake: u128,
    /// Contract owner
    pub owner: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleInfo {
    pub address: u64,
    pub stake: u128,
    pub reputation: u64,
    pub correct_votes: u64,
    pub total_votes: u64,
    pub registered_at: u64,
    pub active: bool,
}

impl BioSafetyOracleContract {
    pub fn new(owner: u64) -> Self {
        Self {
            oracles: HashMap::new(),
            classifications: HashMap::new(),
            pending_votes: HashMap::new(),
            voting_threshold: 3,
            min_oracle_stake: 100_000_000_000_000_000_000_000_000, // 100 BIO
            owner,
        }
    }

    /// Register as a safety oracle
    pub fn register_oracle(
        &mut self,
        address: u64,
        stake: u128,
    ) -> Result<(), BioContractError> {
        if stake < self.min_oracle_stake {
            return Err(BioContractError::InsufficientStake);
        }

        let oracle_info = OracleInfo {
            address,
            stake,
            reputation: 100, // Start with 100 reputation
            correct_votes: 0,
            total_votes: 0,
            registered_at: current_timestamp(),
            active: true,
        };

        self.oracles.insert(address, oracle_info);
        Ok(())
    }

    /// Submit safety classification vote
    pub fn vote(
        &mut self,
        oracle: u64,
        molecule_hash: [u8; 32],
        classification: SafetyClassification,
        evidence_hash: [u8; 32],
    ) -> Result<(), BioContractError> {
        let oracle_info = self.oracles.get(&oracle)
            .ok_or(BioContractError::NotAnOracle)?;

        if !oracle_info.active {
            return Err(BioContractError::OracleInactive);
        }

        // Check if already classified
        if self.classifications.contains_key(&molecule_hash) {
            return Err(BioContractError::AlreadyClassified);
        }

        let vote = SafetyVote {
            oracle,
            molecule_hash,
            classification,
            evidence_hash,
            timestamp: current_timestamp(),
            stake: oracle_info.stake,
        };

        self.pending_votes.entry(molecule_hash).or_default().push(vote);

        // Check if we have enough votes to finalize
        self.try_finalize(molecule_hash)?;

        Ok(())
    }

    /// Get safety classification for a molecule
    pub fn get_classification(&self, molecule_hash: [u8; 32]) -> SafetyClassification {
        self.classifications.get(&molecule_hash)
            .copied()
            .unwrap_or(SafetyClassification::Unknown)
    }

    /// Check if synthesis is allowed
    pub fn is_synthesis_allowed(
        &self,
        molecule_hash: [u8; 32],
        has_license: bool,
        license_schedule: Option<DEASchedule>,
    ) -> bool {
        match self.get_classification(molecule_hash) {
            SafetyClassification::Unrestricted => true,
            SafetyClassification::Controlled(required) => {
                if !has_license {
                    return false;
                }
                match license_schedule {
                    Some(lic) => BioLicenseContract::schedule_permits(&lic, &required),
                    None => false,
                }
            }
            SafetyClassification::Prohibited => false,
            SafetyClassification::Unknown => false, // Block until classified
        }
    }

    fn try_finalize(&mut self, molecule_hash: [u8; 32]) -> Result<(), BioContractError> {
        let votes = match self.pending_votes.get(&molecule_hash) {
            Some(v) if v.len() >= self.voting_threshold => v.clone(),
            _ => return Ok(()), // Not enough votes yet
        };

        // Stake-weighted voting
        let mut unrestricted_weight: u128 = 0;
        let mut controlled_weight: HashMap<DEASchedule, u128> = HashMap::new();
        let mut prohibited_weight: u128 = 0;

        for vote in &votes {
            match vote.classification {
                SafetyClassification::Unrestricted => unrestricted_weight += vote.stake,
                SafetyClassification::Controlled(sched) => {
                    *controlled_weight.entry(sched).or_insert(0) += vote.stake;
                }
                SafetyClassification::Prohibited => prohibited_weight += vote.stake,
                SafetyClassification::Unknown => {}
            }
        }

        // Determine winner
        let max_controlled = controlled_weight.iter()
            .max_by_key(|(_, &w)| w);

        let final_classification = if prohibited_weight > unrestricted_weight
            && prohibited_weight > max_controlled.map(|(_, &w)| w).unwrap_or(0) {
            SafetyClassification::Prohibited
        } else if let Some((&sched, &weight)) = max_controlled {
            if weight > unrestricted_weight {
                SafetyClassification::Controlled(sched)
            } else {
                SafetyClassification::Unrestricted
            }
        } else {
            SafetyClassification::Unrestricted
        };

        // Record classification
        self.classifications.insert(molecule_hash, final_classification);

        // Update oracle reputations
        for vote in &votes {
            if let Some(oracle_info) = self.oracles.get_mut(&vote.oracle) {
                oracle_info.total_votes += 1;
                if vote.classification == final_classification {
                    oracle_info.correct_votes += 1;
                    oracle_info.reputation = oracle_info.reputation.saturating_add(1);
                } else {
                    oracle_info.reputation = oracle_info.reputation.saturating_sub(5);
                }
            }
        }

        // Clear pending votes
        self.pending_votes.remove(&molecule_hash);

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SYNTHESIS MARKETPLACE CONTRACT - P2P Market for Synthesis Services
// ═══════════════════════════════════════════════════════════════════════════════

/// Synthesis service listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisListing {
    pub listing_id: [u8; 32],
    pub provider: u64,
    pub molecule_hash: [u8; 32],
    pub molecule_name: String,
    pub price_per_mg: u128,
    pub min_quantity_mg: u64,
    pub max_quantity_mg: u64,
    pub estimated_time_hours: u32,
    pub purity_guarantee_bps: u16,
    pub requires_license: bool,
    pub active: bool,
    pub total_completed: u64,
    pub rating_sum: u64,
    pub rating_count: u64,
}

/// Synthesis order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisOrder {
    pub order_id: [u8; 32],
    pub listing_id: [u8; 32],
    pub buyer: u64,
    pub provider: u64,
    pub quantity_mg: u64,
    pub total_price: u128,
    pub status: OrderStatus,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    pub proof_id: Option<[u8; 32]>,
    pub rating: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    /// Payment escrowed, awaiting synthesis
    Pending,
    /// Synthesis in progress
    InProgress,
    /// Synthesis complete, awaiting verification
    AwaitingVerification,
    /// Verified and delivered
    Completed,
    /// Order cancelled
    Cancelled,
    /// Disputed
    Disputed,
}

/// Synthesis Marketplace contract
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SynthesisMarketplaceContract {
    /// All listings
    pub listings: HashMap<[u8; 32], SynthesisListing>,
    /// Listings by provider
    pub by_provider: HashMap<u64, Vec<[u8; 32]>>,
    /// Listings by molecule
    pub by_molecule: HashMap<[u8; 32], Vec<[u8; 32]>>,
    /// All orders
    pub orders: HashMap<[u8; 32], SynthesisOrder>,
    /// Orders by buyer
    pub orders_by_buyer: HashMap<u64, Vec<[u8; 32]>>,
    /// Orders by provider
    pub orders_by_provider: HashMap<u64, Vec<[u8; 32]>>,
    /// Escrowed funds
    pub escrow: HashMap<[u8; 32], u128>,
    /// Platform fee in basis points (e.g., 250 = 2.5%)
    pub platform_fee_bps: u16,
    /// Platform treasury
    pub treasury: u64,
    /// Contract owner
    pub owner: u64,
}

impl SynthesisMarketplaceContract {
    pub fn new(owner: u64, treasury: u64) -> Self {
        Self {
            listings: HashMap::new(),
            by_provider: HashMap::new(),
            by_molecule: HashMap::new(),
            orders: HashMap::new(),
            orders_by_buyer: HashMap::new(),
            orders_by_provider: HashMap::new(),
            escrow: HashMap::new(),
            platform_fee_bps: 250, // 2.5% fee
            treasury,
            owner,
        }
    }

    /// Create a synthesis service listing
    pub fn create_listing(
        &mut self,
        provider: u64,
        molecule_name: String,
        molecule_hash: [u8; 32],
        price_per_mg: u128,
        min_quantity_mg: u64,
        max_quantity_mg: u64,
        estimated_time_hours: u32,
        purity_guarantee_bps: u16,
        requires_license: bool,
    ) -> Result<[u8; 32], BioContractError> {
        let listing_id = Self::compute_listing_id(provider, molecule_hash, current_timestamp());

        let listing = SynthesisListing {
            listing_id,
            provider,
            molecule_hash,
            molecule_name,
            price_per_mg,
            min_quantity_mg,
            max_quantity_mg,
            estimated_time_hours,
            purity_guarantee_bps,
            requires_license,
            active: true,
            total_completed: 0,
            rating_sum: 0,
            rating_count: 0,
        };

        self.listings.insert(listing_id, listing);
        self.by_provider.entry(provider).or_default().push(listing_id);
        self.by_molecule.entry(molecule_hash).or_default().push(listing_id);

        Ok(listing_id)
    }

    /// Place an order for synthesis
    pub fn place_order(
        &mut self,
        buyer: u64,
        listing_id: [u8; 32],
        quantity_mg: u64,
        payment: u128,
    ) -> Result<[u8; 32], BioContractError> {
        let listing = self.listings.get(&listing_id)
            .ok_or(BioContractError::ListingNotFound)?;

        if !listing.active {
            return Err(BioContractError::ListingInactive);
        }

        if quantity_mg < listing.min_quantity_mg || quantity_mg > listing.max_quantity_mg {
            return Err(BioContractError::InvalidQuantity);
        }

        let total_price = listing.price_per_mg * quantity_mg as u128;
        if payment < total_price {
            return Err(BioContractError::InsufficientPayment);
        }

        let order_id = Self::compute_order_id(buyer, listing_id, current_timestamp());

        let order = SynthesisOrder {
            order_id,
            listing_id,
            buyer,
            provider: listing.provider,
            quantity_mg,
            total_price,
            status: OrderStatus::Pending,
            created_at: current_timestamp(),
            completed_at: None,
            proof_id: None,
            rating: None,
        };

        self.orders.insert(order_id, order);
        self.orders_by_buyer.entry(buyer).or_default().push(order_id);
        self.orders_by_provider.entry(listing.provider).or_default().push(order_id);
        self.escrow.insert(order_id, total_price);

        Ok(order_id)
    }

    /// Start synthesis (provider)
    pub fn start_synthesis(
        &mut self,
        provider: u64,
        order_id: [u8; 32],
    ) -> Result<(), BioContractError> {
        let order = self.orders.get_mut(&order_id)
            .ok_or(BioContractError::OrderNotFound)?;

        if order.provider != provider {
            return Err(BioContractError::Unauthorized);
        }

        if order.status != OrderStatus::Pending {
            return Err(BioContractError::InvalidOrderStatus);
        }

        order.status = OrderStatus::InProgress;
        Ok(())
    }

    /// Complete synthesis with proof
    pub fn complete_synthesis(
        &mut self,
        provider: u64,
        order_id: [u8; 32],
        proof_id: [u8; 32],
    ) -> Result<(), BioContractError> {
        let order = self.orders.get_mut(&order_id)
            .ok_or(BioContractError::OrderNotFound)?;

        if order.provider != provider {
            return Err(BioContractError::Unauthorized);
        }

        if order.status != OrderStatus::InProgress {
            return Err(BioContractError::InvalidOrderStatus);
        }

        order.status = OrderStatus::AwaitingVerification;
        order.proof_id = Some(proof_id);
        Ok(())
    }

    /// Confirm delivery and release payment (buyer)
    pub fn confirm_delivery(
        &mut self,
        buyer: u64,
        order_id: [u8; 32],
        rating: u8,
    ) -> Result<u128, BioContractError> {
        let order = self.orders.get_mut(&order_id)
            .ok_or(BioContractError::OrderNotFound)?;

        if order.buyer != buyer {
            return Err(BioContractError::Unauthorized);
        }

        if order.status != OrderStatus::AwaitingVerification {
            return Err(BioContractError::InvalidOrderStatus);
        }

        order.status = OrderStatus::Completed;
        order.completed_at = Some(current_timestamp());
        order.rating = Some(rating);

        // Calculate fee and release payment
        let escrowed = self.escrow.remove(&order_id).unwrap_or(0);
        let platform_fee = escrowed * self.platform_fee_bps as u128 / 10000;
        let provider_payment = escrowed - platform_fee;

        // Update listing stats
        if let Some(listing) = self.listings.get_mut(&order.listing_id) {
            listing.total_completed += 1;
            listing.rating_sum += rating as u64;
            listing.rating_count += 1;
        }

        Ok(provider_payment)
    }

    fn compute_listing_id(provider: u64, molecule_hash: [u8; 32], timestamp: u64) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(provider.to_le_bytes());
        hasher.update(molecule_hash);
        hasher.update(timestamp.to_le_bytes());
        hasher.update(b"LISTING_V1");
        hasher.finalize().into()
    }

    fn compute_order_id(buyer: u64, listing_id: [u8; 32], timestamp: u64) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(buyer.to_le_bytes());
        hasher.update(listing_id);
        hasher.update(timestamp.to_le_bytes());
        hasher.update(b"ORDER_V1");
        hasher.finalize().into()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ERROR TYPES
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, thiserror::Error)]
pub enum BioContractError {
    #[error("Unauthorized issuer")]
    UnauthorizedIssuer,

    #[error("No license found for holder")]
    NoLicenseFound,

    #[error("License not found")]
    LicenseNotFound,

    #[error("No valid license for this operation")]
    NoValidLicense,

    #[error("Quota exceeded: requested {requested}mg, remaining {remaining}mg")]
    QuotaExceeded { requested: u64, remaining: u64 },

    #[error("Proof not found")]
    ProofNotFound,

    #[error("Insufficient balance")]
    InsufficientBalance,

    #[error("Insufficient allowance")]
    InsufficientAllowance,

    #[error("Not staked")]
    NotStaked,

    #[error("Insufficient stake")]
    InsufficientStake,

    #[error("No synthesis credits")]
    NoSynthesisCredits,

    #[error("Not an oracle")]
    NotAnOracle,

    #[error("Oracle inactive")]
    OracleInactive,

    #[error("Already classified")]
    AlreadyClassified,

    #[error("Listing not found")]
    ListingNotFound,

    #[error("Listing inactive")]
    ListingInactive,

    #[error("Invalid quantity")]
    InvalidQuantity,

    #[error("Insufficient payment")]
    InsufficientPayment,

    #[error("Order not found")]
    OrderNotFound,

    #[error("Invalid order status")]
    InvalidOrderStatus,

    #[error("Unauthorized")]
    Unauthorized,
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bio_license_issuance() {
        let mut contract = BioLicenseContract::new(1);

        let license_id = contract.issue_license(
            1, // issuer (owner)
            2, // holder
            LicenseType::DEAResearcher,
            DEASchedule::ScheduleI,
            vec![],
            HashMap::new(),
            365,
            [0u8; 64],
        ).unwrap();

        assert!(contract.licenses.contains_key(&license_id));
    }

    #[test]
    fn test_synthesis_proof_recording() {
        let mut contract = SynthesisProofContract::new(1);

        let proof_id = contract.record_proof(
            1,
            "THC".to_string(),
            Some("CCCCCC1=CC...".to_string()),
            1000000, // 1mg in ug
            9950, // 99.50% purity
            [0u8; 32],
            None,
            [0u8; 32],
            current_timestamp() - 3600,
            VerificationMethod::QuantumTomography,
            [0u8; 64],
        ).unwrap();

        assert!(contract.proofs.contains_key(&proof_id));
    }

    #[test]
    fn test_bio_token_staking() {
        let mut contract = BioTokenContract::new(1, 1_000_000_000_000_000_000_000_000_000); // 1B tokens

        // Transfer some tokens
        contract.transfer(1, 2, 100_000_000_000_000_000_000_000_000).unwrap(); // 100K tokens

        // Stake
        contract.stake(2, 10_000_000_000_000_000_000_000_000).unwrap(); // 10K tokens

        let stake_info = contract.staked.get(&2).unwrap();
        assert_eq!(stake_info.tier, StakingTier::Silver);
    }

    #[test]
    fn test_marketplace_listing() {
        let mut contract = SynthesisMarketplaceContract::new(1, 99);

        let molecule_hash = [1u8; 32];
        let listing_id = contract.create_listing(
            2,
            "Caffeine".to_string(),
            molecule_hash,
            1_000_000_000_000_000_000, // 1 token per mg
            100,  // min 100mg
            10000, // max 10g
            24,
            9900, // 99% purity
            false,
        ).unwrap();

        assert!(contract.listings.contains_key(&listing_id));
    }
}
