//! Validator Registry Decentralization Tests
//!
//! Tests to ensure the validator registry properly manages validator lifecycle,
//! stake management, and slashing for Byzantine behavior.
//!
//! CRITICAL SCENARIOS TESTED:
//! 1. Validator registration and activation
//! 2. Stake management (bonding/unbonding)
//! 3. Slashing for Byzantine behavior
//! 4. Validator status transitions
//! 5. Decentralization metrics
//!
//! Run with: cargo test --package q-storage --test validator_registry_tests

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock, atomic::{AtomicU64, Ordering}};
use std::time::{Duration, Instant};

// ============================================================================
// MOCK STRUCTURES FOR VALIDATOR REGISTRY TESTING
// ============================================================================

/// Validator status in the registry
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidatorStatus {
    Pending,    // Applied, waiting for activation
    Active,     // Actively participating in consensus
    Unbonding,  // Requested exit, waiting for unbonding period
    Slashed,    // Slashed for Byzantine behavior
    Exited,     // Successfully exited
}

/// Slashing reason
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlashingReason {
    DoubleSign,        // Signed two different blocks at same height
    DoubleVote,        // Voted for two different proposals in same round
    InvalidProposal,   // Proposed invalid block
    Downtime,          // Missed too many blocks
    CoordinatedAttack, // Participated in multi-validator attack
}

impl SlashingReason {
    /// Get slash percentage (0-100)
    pub fn slash_percentage(&self) -> u64 {
        match self {
            SlashingReason::DoubleSign => 10,        // 10% for double signing
            SlashingReason::DoubleVote => 10,        // 10% for double voting
            SlashingReason::InvalidProposal => 5,    // 5% for invalid proposal
            SlashingReason::Downtime => 1,           // 1% for downtime
            SlashingReason::CoordinatedAttack => 100, // 100% for coordinated attacks
        }
    }
}

/// Validator information
#[derive(Debug, Clone)]
pub struct ValidatorInfo {
    pub address: [u8; 32],
    pub public_key: [u8; 32],
    pub stake: u64,
    pub status: ValidatorStatus,
    pub activation_height: Option<u64>,
    pub exit_height: Option<u64>,
    pub slashed_amount: u64,
    pub blocks_proposed: u64,
    pub blocks_missed: u64,
    pub reputation_score: u64, // 0-100
}

/// Constants for validator registry
const MIN_STAKE: u64 = 32_000_000_000; // 32 QNK (in smallest unit)
const UNBONDING_PERIOD_BLOCKS: u64 = 1_814_400; // ~21 days at 1 block/second
const MAX_VALIDATORS: usize = 1000;
const MIN_VALIDATORS_FOR_CONSENSUS: usize = 4; // 3f+1 with f=1

/// Validator registry managing all validators
pub struct ValidatorRegistry {
    validators: RwLock<HashMap<[u8; 32], ValidatorInfo>>,
    current_height: AtomicU64,
    total_stake: AtomicU64,
    slashing_events: RwLock<Vec<(u64, [u8; 32], SlashingReason, u64)>>, // (height, addr, reason, amount)
}

impl ValidatorRegistry {
    pub fn new() -> Self {
        Self {
            validators: RwLock::new(HashMap::new()),
            current_height: AtomicU64::new(0),
            total_stake: AtomicU64::new(0),
            slashing_events: RwLock::new(Vec::new()),
        }
    }

    pub fn set_height(&self, height: u64) {
        self.current_height.store(height, Ordering::SeqCst);
    }

    /// Register a new validator
    pub fn register_validator(
        &self,
        address: [u8; 32],
        public_key: [u8; 32],
        stake: u64,
    ) -> Result<(), String> {
        // Check minimum stake
        if stake < MIN_STAKE {
            return Err(format!(
                "INSUFFICIENT_STAKE: {} < minimum {}",
                stake, MIN_STAKE
            ));
        }

        let mut validators = self.validators.write().unwrap();

        // Check if already registered
        if validators.contains_key(&address) {
            return Err("ALREADY_REGISTERED: Validator address already exists".to_string());
        }

        // Check max validators
        if validators.len() >= MAX_VALIDATORS {
            return Err(format!(
                "MAX_VALIDATORS: Registry full ({} validators)",
                MAX_VALIDATORS
            ));
        }

        let validator = ValidatorInfo {
            address,
            public_key,
            stake,
            status: ValidatorStatus::Pending,
            activation_height: None,
            exit_height: None,
            slashed_amount: 0,
            blocks_proposed: 0,
            blocks_missed: 0,
            reputation_score: 50, // Start neutral
        };

        validators.insert(address, validator);
        Ok(())
    }

    /// Activate a pending validator
    pub fn activate_validator(&self, address: &[u8; 32]) -> Result<(), String> {
        let mut validators = self.validators.write().unwrap();
        let height = self.current_height.load(Ordering::SeqCst);

        let validator = validators
            .get_mut(address)
            .ok_or("VALIDATOR_NOT_FOUND")?;

        if validator.status != ValidatorStatus::Pending {
            return Err(format!(
                "INVALID_STATUS: Cannot activate validator in {:?} status",
                validator.status
            ));
        }

        validator.status = ValidatorStatus::Active;
        validator.activation_height = Some(height);
        self.total_stake.fetch_add(validator.stake, Ordering::SeqCst);

        Ok(())
    }

    /// Request validator exit (starts unbonding)
    pub fn request_exit(&self, address: &[u8; 32]) -> Result<u64, String> {
        let mut validators = self.validators.write().unwrap();
        let height = self.current_height.load(Ordering::SeqCst);

        let validator = validators
            .get_mut(address)
            .ok_or("VALIDATOR_NOT_FOUND")?;

        if validator.status != ValidatorStatus::Active {
            return Err(format!(
                "INVALID_STATUS: Cannot exit validator in {:?} status",
                validator.status
            ));
        }

        // Check minimum validators
        let active_count = validators
            .values()
            .filter(|v| v.status == ValidatorStatus::Active)
            .count();

        if active_count <= MIN_VALIDATORS_FOR_CONSENSUS {
            return Err(format!(
                "MIN_VALIDATORS: Cannot go below {} active validators",
                MIN_VALIDATORS_FOR_CONSENSUS
            ));
        }

        validator.status = ValidatorStatus::Unbonding;
        validator.exit_height = Some(height + UNBONDING_PERIOD_BLOCKS);
        self.total_stake.fetch_sub(validator.stake, Ordering::SeqCst);

        Ok(height + UNBONDING_PERIOD_BLOCKS)
    }

    /// Complete exit after unbonding period
    pub fn complete_exit(&self, address: &[u8; 32]) -> Result<u64, String> {
        let mut validators = self.validators.write().unwrap();
        let height = self.current_height.load(Ordering::SeqCst);

        let validator = validators
            .get_mut(address)
            .ok_or("VALIDATOR_NOT_FOUND")?;

        if validator.status != ValidatorStatus::Unbonding {
            return Err(format!(
                "INVALID_STATUS: Cannot complete exit for validator in {:?} status",
                validator.status
            ));
        }

        let exit_height = validator.exit_height.ok_or("EXIT_HEIGHT_NOT_SET")?;
        if height < exit_height {
            return Err(format!(
                "UNBONDING_NOT_COMPLETE: {} blocks remaining",
                exit_height - height
            ));
        }

        let refund = validator.stake - validator.slashed_amount;
        validator.status = ValidatorStatus::Exited;

        Ok(refund)
    }

    /// Slash a validator for Byzantine behavior
    pub fn slash_validator(
        &self,
        address: &[u8; 32],
        reason: SlashingReason,
    ) -> Result<u64, String> {
        let mut validators = self.validators.write().unwrap();
        let height = self.current_height.load(Ordering::SeqCst);

        let validator = validators
            .get_mut(address)
            .ok_or("VALIDATOR_NOT_FOUND")?;

        // Can slash active or unbonding validators
        if validator.status != ValidatorStatus::Active
            && validator.status != ValidatorStatus::Unbonding
        {
            return Err(format!(
                "INVALID_STATUS: Cannot slash validator in {:?} status",
                validator.status
            ));
        }

        let slash_percentage = reason.slash_percentage();
        let slash_amount = (validator.stake as u128 * slash_percentage as u128 / 100) as u64;

        validator.slashed_amount += slash_amount;
        validator.reputation_score = validator.reputation_score.saturating_sub(20);

        // For severe violations, immediately slash status
        if slash_percentage >= 10 {
            if validator.status == ValidatorStatus::Active {
                self.total_stake.fetch_sub(validator.stake, Ordering::SeqCst);
            }
            validator.status = ValidatorStatus::Slashed;
        }

        // Record slashing event
        self.slashing_events
            .write()
            .unwrap()
            .push((height, *address, reason, slash_amount));

        Ok(slash_amount)
    }

    /// Get active validator count
    pub fn active_validator_count(&self) -> usize {
        self.validators
            .read()
            .unwrap()
            .values()
            .filter(|v| v.status == ValidatorStatus::Active)
            .count()
    }

    /// Get total staked amount
    pub fn total_staked(&self) -> u64 {
        self.total_stake.load(Ordering::SeqCst)
    }

    /// Get validator info
    pub fn get_validator(&self, address: &[u8; 32]) -> Option<ValidatorInfo> {
        self.validators.read().unwrap().get(address).cloned()
    }

    /// Check if validator set is decentralized enough
    pub fn is_sufficiently_decentralized(&self) -> bool {
        let validators = self.validators.read().unwrap();
        let active: Vec<_> = validators
            .values()
            .filter(|v| v.status == ValidatorStatus::Active)
            .collect();

        if active.len() < MIN_VALIDATORS_FOR_CONSENSUS {
            return false;
        }

        // Check that no single validator has > 33% of stake
        let total: u64 = active.iter().map(|v| v.stake).sum();
        for v in &active {
            if v.stake as u128 * 100 / total as u128 > 33 {
                return false; // Single validator has too much stake
            }
        }

        true
    }

    /// Get slashing history
    pub fn slashing_history(&self) -> Vec<(u64, [u8; 32], SlashingReason, u64)> {
        self.slashing_events.read().unwrap().clone()
    }
}

// ============================================================================
// VALIDATOR REGISTRATION TESTS
// ============================================================================

#[test]
fn test_register_validator_success() {
    let registry = ValidatorRegistry::new();
    let address = [1u8; 32];
    let pubkey = [2u8; 32];

    let result = registry.register_validator(address, pubkey, MIN_STAKE);
    assert!(result.is_ok());

    let validator = registry.get_validator(&address).unwrap();
    assert_eq!(validator.status, ValidatorStatus::Pending);
    assert_eq!(validator.stake, MIN_STAKE);
}

#[test]
fn test_register_validator_insufficient_stake() {
    let registry = ValidatorRegistry::new();
    let address = [1u8; 32];
    let pubkey = [2u8; 32];

    let result = registry.register_validator(address, pubkey, MIN_STAKE - 1);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("INSUFFICIENT_STAKE"));
}

#[test]
fn test_register_duplicate_validator() {
    let registry = ValidatorRegistry::new();
    let address = [1u8; 32];
    let pubkey = [2u8; 32];

    registry.register_validator(address, pubkey, MIN_STAKE).unwrap();

    let result = registry.register_validator(address, pubkey, MIN_STAKE);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("ALREADY_REGISTERED"));
}

// ============================================================================
// VALIDATOR ACTIVATION TESTS
// ============================================================================

#[test]
fn test_activate_validator() {
    let registry = ValidatorRegistry::new();
    let address = [1u8; 32];
    let pubkey = [2u8; 32];

    registry.register_validator(address, pubkey, MIN_STAKE).unwrap();
    registry.set_height(100);
    registry.activate_validator(&address).unwrap();

    let validator = registry.get_validator(&address).unwrap();
    assert_eq!(validator.status, ValidatorStatus::Active);
    assert_eq!(validator.activation_height, Some(100));
    assert_eq!(registry.total_staked(), MIN_STAKE);
}

#[test]
fn test_activate_nonexistent_validator() {
    let registry = ValidatorRegistry::new();
    let address = [1u8; 32];

    let result = registry.activate_validator(&address);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("NOT_FOUND"));
}

#[test]
fn test_double_activation_rejected() {
    let registry = ValidatorRegistry::new();
    let address = [1u8; 32];
    let pubkey = [2u8; 32];

    registry.register_validator(address, pubkey, MIN_STAKE).unwrap();
    registry.activate_validator(&address).unwrap();

    let result = registry.activate_validator(&address);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("INVALID_STATUS"));
}

// ============================================================================
// VALIDATOR EXIT TESTS
// ============================================================================

#[test]
fn test_request_exit() {
    let registry = ValidatorRegistry::new();

    // Register and activate 5 validators to meet minimum
    for i in 0..5 {
        let address = [i as u8; 32];
        let pubkey = [(i + 100) as u8; 32];
        registry.register_validator(address, pubkey, MIN_STAKE).unwrap();
        registry.activate_validator(&address).unwrap();
    }

    registry.set_height(1000);
    let exit_height = registry.request_exit(&[0u8; 32]).unwrap();

    assert_eq!(exit_height, 1000 + UNBONDING_PERIOD_BLOCKS);

    let validator = registry.get_validator(&[0u8; 32]).unwrap();
    assert_eq!(validator.status, ValidatorStatus::Unbonding);
}

#[test]
fn test_exit_below_minimum_validators() {
    let registry = ValidatorRegistry::new();

    // Register exactly MIN_VALIDATORS_FOR_CONSENSUS
    for i in 0..MIN_VALIDATORS_FOR_CONSENSUS {
        let address = [i as u8; 32];
        let pubkey = [(i + 100) as u8; 32];
        registry.register_validator(address, pubkey, MIN_STAKE).unwrap();
        registry.activate_validator(&address).unwrap();
    }

    // Try to exit - should fail
    let result = registry.request_exit(&[0u8; 32]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("MIN_VALIDATORS"));
}

#[test]
fn test_complete_exit_before_unbonding() {
    let registry = ValidatorRegistry::new();

    for i in 0..5 {
        let address = [i as u8; 32];
        let pubkey = [(i + 100) as u8; 32];
        registry.register_validator(address, pubkey, MIN_STAKE).unwrap();
        registry.activate_validator(&address).unwrap();
    }

    registry.set_height(1000);
    registry.request_exit(&[0u8; 32]).unwrap();

    // Try to complete immediately - should fail
    let result = registry.complete_exit(&[0u8; 32]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("UNBONDING_NOT_COMPLETE"));
}

#[test]
fn test_complete_exit_after_unbonding() {
    let registry = ValidatorRegistry::new();

    for i in 0..5 {
        let address = [i as u8; 32];
        let pubkey = [(i + 100) as u8; 32];
        registry.register_validator(address, pubkey, MIN_STAKE).unwrap();
        registry.activate_validator(&address).unwrap();
    }

    registry.set_height(1000);
    let exit_height = registry.request_exit(&[0u8; 32]).unwrap();

    // Fast forward past unbonding
    registry.set_height(exit_height + 1);

    let refund = registry.complete_exit(&[0u8; 32]).unwrap();
    assert_eq!(refund, MIN_STAKE); // Full refund since no slashing

    let validator = registry.get_validator(&[0u8; 32]).unwrap();
    assert_eq!(validator.status, ValidatorStatus::Exited);
}

// ============================================================================
// SLASHING TESTS
// ============================================================================

#[test]
fn test_slash_for_double_sign() {
    let registry = ValidatorRegistry::new();
    let address = [1u8; 32];
    let pubkey = [2u8; 32];

    registry.register_validator(address, pubkey, MIN_STAKE).unwrap();
    registry.activate_validator(&address).unwrap();

    let slash_amount = registry.slash_validator(&address, SlashingReason::DoubleSign).unwrap();

    // Should slash 10%
    assert_eq!(slash_amount, MIN_STAKE / 10);

    let validator = registry.get_validator(&address).unwrap();
    assert_eq!(validator.status, ValidatorStatus::Slashed);
    assert_eq!(validator.slashed_amount, slash_amount);
}

#[test]
fn test_slash_for_coordinated_attack() {
    let registry = ValidatorRegistry::new();
    let address = [1u8; 32];
    let pubkey = [2u8; 32];

    registry.register_validator(address, pubkey, MIN_STAKE).unwrap();
    registry.activate_validator(&address).unwrap();

    let slash_amount = registry.slash_validator(&address, SlashingReason::CoordinatedAttack).unwrap();

    // Should slash 100%
    assert_eq!(slash_amount, MIN_STAKE);

    let validator = registry.get_validator(&address).unwrap();
    assert_eq!(validator.status, ValidatorStatus::Slashed);
}

#[test]
fn test_slash_reduces_reputation() {
    let registry = ValidatorRegistry::new();
    let address = [1u8; 32];
    let pubkey = [2u8; 32];

    registry.register_validator(address, pubkey, MIN_STAKE).unwrap();
    registry.activate_validator(&address).unwrap();

    let initial_rep = registry.get_validator(&address).unwrap().reputation_score;
    registry.slash_validator(&address, SlashingReason::Downtime).unwrap();

    let final_rep = registry.get_validator(&address).unwrap().reputation_score;
    assert!(final_rep < initial_rep);
}

#[test]
fn test_slashing_history_recorded() {
    let registry = ValidatorRegistry::new();

    for i in 0..3 {
        let address = [i as u8; 32];
        let pubkey = [(i + 100) as u8; 32];
        registry.register_validator(address, pubkey, MIN_STAKE).unwrap();
        registry.activate_validator(&address).unwrap();
    }

    registry.set_height(100);
    registry.slash_validator(&[0u8; 32], SlashingReason::DoubleSign).unwrap();
    registry.set_height(200);
    registry.slash_validator(&[1u8; 32], SlashingReason::DoubleVote).unwrap();

    let history = registry.slashing_history();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].0, 100); // First slashing at height 100
    assert_eq!(history[1].0, 200); // Second at height 200
}

// ============================================================================
// DECENTRALIZATION TESTS
// ============================================================================

#[test]
fn test_decentralization_check_insufficient_validators() {
    let registry = ValidatorRegistry::new();

    // Only 3 validators (below minimum of 4)
    for i in 0..3 {
        let address = [i as u8; 32];
        let pubkey = [(i + 100) as u8; 32];
        registry.register_validator(address, pubkey, MIN_STAKE).unwrap();
        registry.activate_validator(&address).unwrap();
    }

    assert!(!registry.is_sufficiently_decentralized());
}

#[test]
fn test_decentralization_check_stake_concentration() {
    let registry = ValidatorRegistry::new();

    // One whale with 50% stake
    let whale_stake = MIN_STAKE * 10;
    let small_stake = MIN_STAKE;

    registry.register_validator([0u8; 32], [100u8; 32], whale_stake).unwrap();
    registry.activate_validator(&[0u8; 32]).unwrap();

    for i in 1..5 {
        let address = [i as u8; 32];
        let pubkey = [(i + 100) as u8; 32];
        registry.register_validator(address, pubkey, small_stake).unwrap();
        registry.activate_validator(&address).unwrap();
    }

    // Whale has > 33% of stake
    assert!(!registry.is_sufficiently_decentralized());
}

#[test]
fn test_decentralization_check_balanced() {
    let registry = ValidatorRegistry::new();

    // 10 validators with equal stake
    for i in 0..10 {
        let address = [i as u8; 32];
        let pubkey = [(i + 100) as u8; 32];
        registry.register_validator(address, pubkey, MIN_STAKE).unwrap();
        registry.activate_validator(&address).unwrap();
    }

    assert!(registry.is_sufficiently_decentralized());
}

// ============================================================================
// STRESS TESTS
// ============================================================================

#[test]
fn test_many_validators() {
    let registry = ValidatorRegistry::new();

    // Register 100 validators
    for i in 0..100 {
        let mut address = [0u8; 32];
        address[0] = (i / 256) as u8;
        address[1] = (i % 256) as u8;

        let mut pubkey = [0u8; 32];
        pubkey[0] = ((i + 100) / 256) as u8;
        pubkey[1] = ((i + 100) % 256) as u8;

        registry.register_validator(address, pubkey, MIN_STAKE).unwrap();
        registry.activate_validator(&address).unwrap();
    }

    assert_eq!(registry.active_validator_count(), 100);
    assert_eq!(registry.total_staked(), MIN_STAKE * 100);
}

#[test]
fn test_exit_refund_after_partial_slash() {
    let registry = ValidatorRegistry::new();

    for i in 0..5 {
        let address = [i as u8; 32];
        let pubkey = [(i + 100) as u8; 32];
        registry.register_validator(address, pubkey, MIN_STAKE).unwrap();
        registry.activate_validator(&address).unwrap();
    }

    // Slash for downtime (1%)
    registry.slash_validator(&[0u8; 32], SlashingReason::Downtime).unwrap();

    // The validator status doesn't change to Slashed for minor offenses
    let validator = registry.get_validator(&[0u8; 32]).unwrap();
    assert_eq!(validator.status, ValidatorStatus::Active); // Still active for minor slash

    // Request exit
    registry.request_exit(&[0u8; 32]).unwrap();

    // Fast forward
    registry.set_height(UNBONDING_PERIOD_BLOCKS + 1);

    let refund = registry.complete_exit(&[0u8; 32]).unwrap();
    let expected_refund = MIN_STAKE - (MIN_STAKE / 100); // 99% refund
    assert_eq!(refund, expected_refund);
}
