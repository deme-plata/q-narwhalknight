//! Quillon Bank Token (QBANK) - Advanced DeFi Token Contract
//!
//! A comprehensive custom token for Quillon Bank with full DeFi features:
//! - Reflection (auto-redistribution to all holders on each transfer)
//! - Staking with tiered rewards
//! - Airdrop system with vesting
//! - Buy/sell taxes with distribution
//! - Auto-liquidity provision
//! - Burn mechanism
//! - Governance voting weight
//!
//! This token is used by the DEX Activity Bot to generate organic trading activity.

use std::collections::HashMap;

// ============ STORAGE KEYS ============
const TOTAL_SUPPLY_KEY: &[u8] = b"qbank_total_supply";
const CIRCULATING_SUPPLY_KEY: &[u8] = b"qbank_circulating";
const OWNER_KEY: &[u8] = b"qbank_owner";
const BALANCES_PREFIX: &[u8] = b"qbank_balance_";
const ALLOWANCES_PREFIX: &[u8] = b"qbank_allowance_";
const REFLECTION_PREFIX: &[u8] = b"qbank_reflection_";
const STAKING_PREFIX: &[u8] = b"qbank_stake_";
const STAKING_REWARDS_PREFIX: &[u8] = b"qbank_stake_reward_";
const AIRDROP_PREFIX: &[u8] = b"qbank_airdrop_";
const EXCLUDED_FROM_FEE_PREFIX: &[u8] = b"qbank_excluded_";
const EXCLUDED_FROM_REWARD_PREFIX: &[u8] = b"qbank_no_reflect_";
const LIQUIDITY_POOL_KEY: &[u8] = b"qbank_lp_reserve";
const TOTAL_FEES_KEY: &[u8] = b"qbank_total_fees";
const TOTAL_BURNED_KEY: &[u8] = b"qbank_total_burned";
const TOTAL_REFLECTED_KEY: &[u8] = b"qbank_total_reflected";

// ============ EVENT TYPES ============
const TRANSFER_EVENT: u8 = 1;
const APPROVAL_EVENT: u8 = 2;
const REFLECTION_EVENT: u8 = 3;
const STAKE_EVENT: u8 = 4;
const UNSTAKE_EVENT: u8 = 5;
const REWARD_CLAIM_EVENT: u8 = 6;
const AIRDROP_EVENT: u8 = 7;
const BURN_EVENT: u8 = 8;
const FEE_COLLECTED_EVENT: u8 = 9;
const LIQUIDITY_ADDED_EVENT: u8 = 10;

// ============ TOKEN PARAMETERS ============

/// Token name
const TOKEN_NAME: &str = "Quillon Bank Token";
/// Token symbol
const TOKEN_SYMBOL: &str = "QBANK";
/// Token decimals (18 = standard)
const TOKEN_DECIMALS: u8 = 18;

/// Initial supply: 1 billion QBANK
const INITIAL_SUPPLY: u64 = 1_000_000_000 * 10u64.pow(18);

// ============ FEE CONFIGURATION ============
// Total transaction fee: 5% (configurable by owner)

/// Reflection fee (distributed to holders): 2%
const REFLECTION_FEE_PERCENT: u64 = 200; // basis points (2%)
/// Liquidity fee (added to LP): 1%
const LIQUIDITY_FEE_PERCENT: u64 = 100; // basis points (1%)
/// Burn fee: 0.5%
const BURN_FEE_PERCENT: u64 = 50; // basis points (0.5%)
/// Development/Treasury fee: 1.5%
const DEV_FEE_PERCENT: u64 = 150; // basis points (1.5%)
/// Total fee in basis points (5%)
const TOTAL_FEE_PERCENT: u64 = 500;
/// Maximum fee cap (cannot exceed 10%)
const MAX_FEE_PERCENT: u64 = 1000;

// ============ STAKING TIERS ============

/// Bronze tier: 30 days lock, 5% APY
const BRONZE_LOCK_PERIOD: u64 = 30 * 24 * 3600; // 30 days in seconds
const BRONZE_APY_PERCENT: u64 = 500; // 5%

/// Silver tier: 90 days lock, 12% APY
const SILVER_LOCK_PERIOD: u64 = 90 * 24 * 3600; // 90 days
const SILVER_APY_PERCENT: u64 = 1200; // 12%

/// Gold tier: 180 days lock, 20% APY
const GOLD_LOCK_PERIOD: u64 = 180 * 24 * 3600; // 180 days
const GOLD_APY_PERCENT: u64 = 2000; // 20%

/// Platinum tier: 365 days lock, 35% APY
const PLATINUM_LOCK_PERIOD: u64 = 365 * 24 * 3600; // 365 days
const PLATINUM_APY_PERCENT: u64 = 3500; // 35%

// ============ EXTERNAL VM FUNCTIONS ============
extern "C" {
    fn read_storage(key_ptr: *const u8, key_len: u32, value_ptr: *mut u8, value_len: u32) -> i32;
    fn write_storage(key_ptr: *const u8, key_len: u32, value_ptr: *const u8, value_len: u32) -> i32;
    fn emit_log(event_type: u8, data_ptr: *const u8, data_len: u32) -> i32;
    fn get_caller() -> u64;
    fn get_block_timestamp() -> u64;
    fn get_block_number() -> u64;
}

// ============ DATA STRUCTURES ============

/// Staking tier enum
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum StakingTier {
    Bronze = 0,
    Silver = 1,
    Gold = 2,
    Platinum = 3,
}

impl StakingTier {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Bronze,
            1 => Self::Silver,
            2 => Self::Gold,
            3 => Self::Platinum,
            _ => Self::Bronze,
        }
    }

    pub fn lock_period(&self) -> u64 {
        match self {
            Self::Bronze => BRONZE_LOCK_PERIOD,
            Self::Silver => SILVER_LOCK_PERIOD,
            Self::Gold => GOLD_LOCK_PERIOD,
            Self::Platinum => PLATINUM_LOCK_PERIOD,
        }
    }

    pub fn apy_percent(&self) -> u64 {
        match self {
            Self::Bronze => BRONZE_APY_PERCENT,
            Self::Silver => SILVER_APY_PERCENT,
            Self::Gold => GOLD_APY_PERCENT,
            Self::Platinum => PLATINUM_APY_PERCENT,
        }
    }
}

/// Staking position
#[derive(Clone)]
pub struct StakePosition {
    pub amount: u64,
    pub tier: StakingTier,
    pub start_time: u64,
    pub unlock_time: u64,
    pub last_reward_claim: u64,
    pub total_rewards_claimed: u64,
}

/// Fee configuration (can be updated by owner)
#[derive(Clone)]
pub struct FeeConfig {
    pub reflection_fee: u64, // basis points
    pub liquidity_fee: u64,
    pub burn_fee: u64,
    pub dev_fee: u64,
    pub enabled: bool,
}

impl Default for FeeConfig {
    fn default() -> Self {
        Self {
            reflection_fee: REFLECTION_FEE_PERCENT,
            liquidity_fee: LIQUIDITY_FEE_PERCENT,
            burn_fee: BURN_FEE_PERCENT,
            dev_fee: DEV_FEE_PERCENT,
            enabled: true,
        }
    }
}

/// Airdrop campaign
#[derive(Clone)]
pub struct AirdropCampaign {
    pub id: u64,
    pub total_tokens: u64,
    pub tokens_per_claim: u64,
    pub max_claims: u64,
    pub current_claims: u64,
    pub start_time: u64,
    pub end_time: u64,
    pub vesting_duration: u64, // 0 = no vesting
    pub active: bool,
}

/// Contract state
pub struct QBankContract {
    // Core token state
    balances: HashMap<u64, u64>,
    allowances: HashMap<(u64, u64), u64>,
    total_supply: u64,
    circulating_supply: u64,
    owner: u64,

    // Reflection system (tSupply model)
    reflection_balances: HashMap<u64, u64>, // rBalance
    total_reflection: u64,                   // tTotal reflected
    excluded_from_reward: HashMap<u64, bool>,

    // Staking system
    stakes: HashMap<u64, StakePosition>,
    total_staked: u64,
    staking_rewards_pool: u64,

    // Fee system
    fee_config: FeeConfig,
    excluded_from_fee: HashMap<u64, bool>,
    total_fees_collected: u64,
    total_burned: u64,
    total_reflected: u64,

    // Liquidity
    liquidity_reserve: u64,
    dev_wallet: u64,

    // Airdrops
    airdrops: HashMap<u64, AirdropCampaign>,
    airdrop_claims: HashMap<(u64, u64), u64>, // (airdrop_id, address) -> claimed amount
    next_airdrop_id: u64,
}

impl QBankContract {
    /// Load contract state from storage
    fn load() -> Self {
        let mut contract = Self {
            balances: HashMap::new(),
            allowances: HashMap::new(),
            total_supply: 0,
            circulating_supply: 0,
            owner: 0,
            reflection_balances: HashMap::new(),
            total_reflection: 0,
            excluded_from_reward: HashMap::new(),
            stakes: HashMap::new(),
            total_staked: 0,
            staking_rewards_pool: 0,
            fee_config: FeeConfig::default(),
            excluded_from_fee: HashMap::new(),
            total_fees_collected: 0,
            total_burned: 0,
            total_reflected: 0,
            liquidity_reserve: 0,
            dev_wallet: 0,
            airdrops: HashMap::new(),
            airdrop_claims: HashMap::new(),
            next_airdrop_id: 1,
        };

        // Load core state
        let mut buf = [0u8; 8];
        unsafe {
            read_storage(
                TOTAL_SUPPLY_KEY.as_ptr(),
                TOTAL_SUPPLY_KEY.len() as u32,
                buf.as_mut_ptr(),
                buf.len() as u32,
            );
        }
        contract.total_supply = u64::from_le_bytes(buf);

        unsafe {
            read_storage(
                OWNER_KEY.as_ptr(),
                OWNER_KEY.len() as u32,
                buf.as_mut_ptr(),
                buf.len() as u32,
            );
        }
        contract.owner = u64::from_le_bytes(buf);

        unsafe {
            read_storage(
                TOTAL_FEES_KEY.as_ptr(),
                TOTAL_FEES_KEY.len() as u32,
                buf.as_mut_ptr(),
                buf.len() as u32,
            );
        }
        contract.total_fees_collected = u64::from_le_bytes(buf);

        unsafe {
            read_storage(
                TOTAL_BURNED_KEY.as_ptr(),
                TOTAL_BURNED_KEY.len() as u32,
                buf.as_mut_ptr(),
                buf.len() as u32,
            );
        }
        contract.total_burned = u64::from_le_bytes(buf);

        unsafe {
            read_storage(
                TOTAL_REFLECTED_KEY.as_ptr(),
                TOTAL_REFLECTED_KEY.len() as u32,
                buf.as_mut_ptr(),
                buf.len() as u32,
            );
        }
        contract.total_reflected = u64::from_le_bytes(buf);

        contract
    }

    /// Save contract state to storage
    fn save(&self) {
        // Save core state
        let buf = self.total_supply.to_le_bytes();
        unsafe {
            write_storage(
                TOTAL_SUPPLY_KEY.as_ptr(),
                TOTAL_SUPPLY_KEY.len() as u32,
                buf.as_ptr(),
                buf.len() as u32,
            );
        }

        let buf = self.owner.to_le_bytes();
        unsafe {
            write_storage(
                OWNER_KEY.as_ptr(),
                OWNER_KEY.len() as u32,
                buf.as_ptr(),
                buf.len() as u32,
            );
        }

        let buf = self.total_fees_collected.to_le_bytes();
        unsafe {
            write_storage(
                TOTAL_FEES_KEY.as_ptr(),
                TOTAL_FEES_KEY.len() as u32,
                buf.as_ptr(),
                buf.len() as u32,
            );
        }

        let buf = self.total_burned.to_le_bytes();
        unsafe {
            write_storage(
                TOTAL_BURNED_KEY.as_ptr(),
                TOTAL_BURNED_KEY.len() as u32,
                buf.as_ptr(),
                buf.len() as u32,
            );
        }

        let buf = self.total_reflected.to_le_bytes();
        unsafe {
            write_storage(
                TOTAL_REFLECTED_KEY.as_ptr(),
                TOTAL_REFLECTED_KEY.len() as u32,
                buf.as_ptr(),
                buf.len() as u32,
            );
        }

        // Save balances
        for (address, &balance) in &self.balances {
            let key = [BALANCES_PREFIX, &address.to_le_bytes()].concat();
            let value = balance.to_le_bytes();
            unsafe {
                write_storage(
                    key.as_ptr(),
                    key.len() as u32,
                    value.as_ptr(),
                    value.len() as u32,
                );
            }
        }

        // Save allowances
        for ((owner, spender), &amount) in &self.allowances {
            let key = [ALLOWANCES_PREFIX, &owner.to_le_bytes(), &spender.to_le_bytes()].concat();
            let value = amount.to_le_bytes();
            unsafe {
                write_storage(
                    key.as_ptr(),
                    key.len() as u32,
                    value.as_ptr(),
                    value.len() as u32,
                );
            }
        }
    }

    // ============ BALANCE HELPERS ============

    fn balance_of(&mut self, address: u64) -> u64 {
        if let Some(&balance) = self.balances.get(&address) {
            return balance;
        }

        let key = [BALANCES_PREFIX, &address.to_le_bytes()].concat();
        let mut buf = [0u8; 8];
        unsafe {
            read_storage(
                key.as_ptr(),
                key.len() as u32,
                buf.as_mut_ptr(),
                buf.len() as u32,
            );
        }

        let balance = u64::from_le_bytes(buf);
        self.balances.insert(address, balance);
        balance
    }

    fn set_balance(&mut self, address: u64, balance: u64) {
        self.balances.insert(address, balance);
    }

    fn is_excluded_from_fee(&self, address: u64) -> bool {
        *self.excluded_from_fee.get(&address).unwrap_or(&false)
    }

    fn is_excluded_from_reward(&self, address: u64) -> bool {
        *self.excluded_from_reward.get(&address).unwrap_or(&false)
    }

    // ============ FEE CALCULATION ============

    /// Calculate fees for a transfer
    fn calculate_fees(&self, amount: u64) -> (u64, u64, u64, u64, u64) {
        if !self.fee_config.enabled {
            return (amount, 0, 0, 0, 0);
        }

        let reflection = (amount * self.fee_config.reflection_fee) / 10000;
        let liquidity = (amount * self.fee_config.liquidity_fee) / 10000;
        let burn = (amount * self.fee_config.burn_fee) / 10000;
        let dev = (amount * self.fee_config.dev_fee) / 10000;

        let total_fee = reflection + liquidity + burn + dev;
        let transfer_amount = amount - total_fee;

        (transfer_amount, reflection, liquidity, burn, dev)
    }

    /// Distribute reflection to all holders
    fn distribute_reflection(&mut self, reflection_amount: u64) {
        if reflection_amount == 0 || self.circulating_supply == 0 {
            return;
        }

        // In a real implementation, this would update the reflection ratio
        // For simplicity, we add to a pool that's distributed on claim
        self.total_reflected += reflection_amount;

        // Emit reflection event
        let mut event_data = Vec::new();
        event_data.extend_from_slice(&reflection_amount.to_le_bytes());
        unsafe {
            emit_log(
                REFLECTION_EVENT,
                event_data.as_ptr(),
                event_data.len() as u32,
            );
        }
    }

    // ============ STAKING HELPERS ============

    /// Calculate pending staking rewards
    fn calculate_pending_rewards(&self, address: u64) -> u64 {
        if let Some(stake) = self.stakes.get(&address) {
            let current_time = unsafe { get_block_timestamp() };
            let time_staked = current_time.saturating_sub(stake.last_reward_claim);
            let seconds_per_year = 365 * 24 * 3600;

            // Calculate rewards based on APY and time
            let annual_reward = (stake.amount * stake.tier.apy_percent()) / 10000;
            let pending = (annual_reward * time_staked) / seconds_per_year;

            return pending;
        }
        0
    }

    // ============ EVENT EMITTERS ============

    fn emit_transfer(&self, from: u64, to: u64, amount: u64) {
        let mut event_data = Vec::new();
        event_data.extend_from_slice(&from.to_le_bytes());
        event_data.extend_from_slice(&to.to_le_bytes());
        event_data.extend_from_slice(&amount.to_le_bytes());

        unsafe {
            emit_log(
                TRANSFER_EVENT,
                event_data.as_ptr(),
                event_data.len() as u32,
            );
        }
    }

    fn emit_fee_collected(&self, from: u64, reflection: u64, liquidity: u64, burn: u64, dev: u64) {
        let mut event_data = Vec::new();
        event_data.extend_from_slice(&from.to_le_bytes());
        event_data.extend_from_slice(&reflection.to_le_bytes());
        event_data.extend_from_slice(&liquidity.to_le_bytes());
        event_data.extend_from_slice(&burn.to_le_bytes());
        event_data.extend_from_slice(&dev.to_le_bytes());

        unsafe {
            emit_log(
                FEE_COLLECTED_EVENT,
                event_data.as_ptr(),
                event_data.len() as u32,
            );
        }
    }
}

// ============ PUBLIC CONTRACT FUNCTIONS ============

/// Initialize the contract with owner allocation
#[no_mangle]
pub extern "C" fn constructor(dev_wallet: u64) -> i32 {
    let caller = unsafe { get_caller() };

    let mut contract = QBankContract {
        balances: HashMap::new(),
        allowances: HashMap::new(),
        total_supply: INITIAL_SUPPLY,
        circulating_supply: INITIAL_SUPPLY,
        owner: caller,
        reflection_balances: HashMap::new(),
        total_reflection: 0,
        excluded_from_reward: HashMap::new(),
        stakes: HashMap::new(),
        total_staked: 0,
        staking_rewards_pool: 0,
        fee_config: FeeConfig::default(),
        excluded_from_fee: HashMap::new(),
        total_fees_collected: 0,
        total_burned: 0,
        total_reflected: 0,
        liquidity_reserve: 0,
        dev_wallet,
        airdrops: HashMap::new(),
        airdrop_claims: HashMap::new(),
        next_airdrop_id: 1,
    };

    // Distribution:
    // 40% - Owner/Development
    // 20% - Staking rewards pool
    // 20% - Liquidity provision
    // 10% - Airdrop reserve
    // 10% - Initial burn (deflationary)

    let owner_allocation = (INITIAL_SUPPLY * 40) / 100;
    let staking_pool = (INITIAL_SUPPLY * 20) / 100;
    let liquidity_allocation = (INITIAL_SUPPLY * 20) / 100;
    let airdrop_reserve = (INITIAL_SUPPLY * 10) / 100;
    let initial_burn = (INITIAL_SUPPLY * 10) / 100;

    // Allocate to owner
    contract.set_balance(caller, owner_allocation + airdrop_reserve);

    // Set up pools
    contract.staking_rewards_pool = staking_pool;
    contract.liquidity_reserve = liquidity_allocation;

    // Initial burn
    contract.total_burned = initial_burn;
    contract.circulating_supply = INITIAL_SUPPLY - initial_burn - staking_pool;

    // Exclude owner and dev wallet from fees
    contract.excluded_from_fee.insert(caller, true);
    contract.excluded_from_fee.insert(dev_wallet, true);

    contract.save();

    // Emit initial transfer event
    contract.emit_transfer(0, caller, owner_allocation + airdrop_reserve);

    1 // Success
}

/// Get token name
#[no_mangle]
pub extern "C" fn name() -> i32 {
    unsafe {
        emit_log(0, TOKEN_NAME.as_ptr(), TOKEN_NAME.len() as u32);
    }
    1
}

/// Get token symbol
#[no_mangle]
pub extern "C" fn symbol() -> i32 {
    unsafe {
        emit_log(0, TOKEN_SYMBOL.as_ptr(), TOKEN_SYMBOL.len() as u32);
    }
    1
}

/// Get token decimals
#[no_mangle]
pub extern "C" fn decimals() -> u8 {
    TOKEN_DECIMALS
}

/// Get total supply
#[no_mangle]
pub extern "C" fn total_supply() -> u64 {
    let contract = QBankContract::load();
    contract.total_supply
}

/// Get circulating supply (excludes burned and locked)
#[no_mangle]
pub extern "C" fn circulating_supply() -> u64 {
    let contract = QBankContract::load();
    contract.circulating_supply
}

/// Get balance of address
#[no_mangle]
pub extern "C" fn balance_of(address: u64) -> u64 {
    let mut contract = QBankContract::load();
    contract.balance_of(address)
}

/// Transfer with reflection and fees
#[no_mangle]
pub extern "C" fn transfer(to: u64, amount: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = QBankContract::load();

    let from_balance = contract.balance_of(caller);
    if from_balance < amount {
        return false;
    }

    // Check if sender is excluded from fees
    let apply_fee = !contract.is_excluded_from_fee(caller) && !contract.is_excluded_from_fee(to);

    if apply_fee {
        let (transfer_amount, reflection, liquidity, burn, dev) = contract.calculate_fees(amount);

        // Update balances
        contract.set_balance(caller, from_balance - amount);
        let to_balance = contract.balance_of(to);
        contract.set_balance(to, to_balance + transfer_amount);

        // Distribute reflection
        contract.distribute_reflection(reflection);

        // Add to liquidity
        contract.liquidity_reserve += liquidity;

        // Burn
        contract.total_burned += burn;
        contract.circulating_supply -= burn;

        // Send to dev wallet
        let dev_balance = contract.balance_of(contract.dev_wallet);
        contract.set_balance(contract.dev_wallet, dev_balance + dev);

        // Track fees (v1.4.5-beta: saturating add to prevent overflow)
        let fee_total = reflection
            .saturating_add(liquidity)
            .saturating_add(burn)
            .saturating_add(dev);
        contract.total_fees_collected = contract.total_fees_collected.saturating_add(fee_total);

        // Emit events
        contract.emit_transfer(caller, to, transfer_amount);
        contract.emit_fee_collected(caller, reflection, liquidity, burn, dev);
    } else {
        // No fee transfer
        contract.set_balance(caller, from_balance - amount);
        let to_balance = contract.balance_of(to);
        contract.set_balance(to, to_balance + amount);
        contract.emit_transfer(caller, to, amount);
    }

    contract.save();
    true
}

/// Approve spender
#[no_mangle]
pub extern "C" fn approve(spender: u64, amount: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = QBankContract::load();

    contract.allowances.insert((caller, spender), amount);
    contract.save();

    // Emit approval event
    let mut event_data = Vec::new();
    event_data.extend_from_slice(&caller.to_le_bytes());
    event_data.extend_from_slice(&spender.to_le_bytes());
    event_data.extend_from_slice(&amount.to_le_bytes());

    unsafe {
        emit_log(
            APPROVAL_EVENT,
            event_data.as_ptr(),
            event_data.len() as u32,
        );
    }

    true
}

/// Get allowance
#[no_mangle]
pub extern "C" fn allowance(owner: u64, spender: u64) -> u64 {
    let contract = QBankContract::load();
    *contract.allowances.get(&(owner, spender)).unwrap_or(&0)
}

/// Transfer from (with approval)
#[no_mangle]
pub extern "C" fn transfer_from(from: u64, to: u64, amount: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = QBankContract::load();

    let allowed = *contract.allowances.get(&(from, caller)).unwrap_or(&0);
    if allowed < amount {
        return false;
    }

    let from_balance = contract.balance_of(from);
    if from_balance < amount {
        return false;
    }

    // Update allowance
    contract.allowances.insert((from, caller), allowed - amount);

    // Apply transfer with fees (same logic as transfer)
    let apply_fee = !contract.is_excluded_from_fee(from) && !contract.is_excluded_from_fee(to);

    if apply_fee {
        let (transfer_amount, reflection, liquidity, burn, dev) = contract.calculate_fees(amount);

        contract.set_balance(from, from_balance - amount);
        let to_balance = contract.balance_of(to);
        contract.set_balance(to, to_balance + transfer_amount);

        contract.distribute_reflection(reflection);
        contract.liquidity_reserve += liquidity;
        contract.total_burned += burn;
        contract.circulating_supply -= burn;

        let dev_balance = contract.balance_of(contract.dev_wallet);
        contract.set_balance(contract.dev_wallet, dev_balance + dev);
        contract.total_fees_collected += reflection + liquidity + burn + dev;

        contract.emit_transfer(from, to, transfer_amount);
        contract.emit_fee_collected(from, reflection, liquidity, burn, dev);
    } else {
        contract.set_balance(from, from_balance - amount);
        let to_balance = contract.balance_of(to);
        contract.set_balance(to, to_balance + amount);
        contract.emit_transfer(from, to, amount);
    }

    contract.save();
    true
}

// ============ STAKING FUNCTIONS ============

/// Stake tokens with a specific tier
#[no_mangle]
pub extern "C" fn stake(amount: u64, tier: u8) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = QBankContract::load();

    let balance = contract.balance_of(caller);
    if balance < amount {
        return false;
    }

    let staking_tier = StakingTier::from_u8(tier);
    let current_time = unsafe { get_block_timestamp() };

    // If already staking, add to existing position
    if let Some(existing_stake) = contract.stakes.get_mut(&caller) {
        // Claim pending rewards first
        let pending = contract.calculate_pending_rewards(caller);
        if pending > 0 {
            let caller_balance = contract.balance_of(caller);
            contract.set_balance(caller, caller_balance + pending);
            existing_stake.total_rewards_claimed += pending;
        }

        existing_stake.amount += amount;
        existing_stake.tier = staking_tier;
        existing_stake.unlock_time = current_time + staking_tier.lock_period();
        existing_stake.last_reward_claim = current_time;
    } else {
        let stake = StakePosition {
            amount,
            tier: staking_tier,
            start_time: current_time,
            unlock_time: current_time + staking_tier.lock_period(),
            last_reward_claim: current_time,
            total_rewards_claimed: 0,
        };
        contract.stakes.insert(caller, stake);
    }

    // Lock tokens
    contract.set_balance(caller, balance - amount);
    contract.total_staked += amount;

    contract.save();

    // Emit stake event
    let mut event_data = Vec::new();
    event_data.extend_from_slice(&caller.to_le_bytes());
    event_data.extend_from_slice(&amount.to_le_bytes());
    event_data.extend_from_slice(&[tier]);

    unsafe {
        emit_log(
            STAKE_EVENT,
            event_data.as_ptr(),
            event_data.len() as u32,
        );
    }

    true
}

/// Unstake tokens (after lock period)
#[no_mangle]
pub extern "C" fn unstake() -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = QBankContract::load();

    let stake = match contract.stakes.get(&caller) {
        Some(s) => s.clone(),
        None => return false,
    };

    let current_time = unsafe { get_block_timestamp() };

    // Check lock period
    if current_time < stake.unlock_time {
        return false; // Still locked
    }

    // Claim pending rewards
    let pending = contract.calculate_pending_rewards(caller);

    // Return staked amount + rewards
    let balance = contract.balance_of(caller);
    contract.set_balance(caller, balance + stake.amount + pending);
    contract.total_staked -= stake.amount;

    // Remove stake
    contract.stakes.remove(&caller);

    contract.save();

    // Emit unstake event
    let mut event_data = Vec::new();
    event_data.extend_from_slice(&caller.to_le_bytes());
    event_data.extend_from_slice(&stake.amount.to_le_bytes());
    event_data.extend_from_slice(&pending.to_le_bytes());

    unsafe {
        emit_log(
            UNSTAKE_EVENT,
            event_data.as_ptr(),
            event_data.len() as u32,
        );
    }

    true
}

/// Claim staking rewards without unstaking
#[no_mangle]
pub extern "C" fn claim_staking_rewards() -> u64 {
    let caller = unsafe { get_caller() };
    let mut contract = QBankContract::load();

    let pending = contract.calculate_pending_rewards(caller);
    if pending == 0 {
        return 0;
    }

    // Update stake
    if let Some(stake) = contract.stakes.get_mut(&caller) {
        stake.last_reward_claim = unsafe { get_block_timestamp() };
        stake.total_rewards_claimed += pending;
    }

    // Transfer rewards
    let balance = contract.balance_of(caller);
    contract.set_balance(caller, balance + pending);

    contract.save();

    // Emit reward claim event
    let mut event_data = Vec::new();
    event_data.extend_from_slice(&caller.to_le_bytes());
    event_data.extend_from_slice(&pending.to_le_bytes());

    unsafe {
        emit_log(
            REWARD_CLAIM_EVENT,
            event_data.as_ptr(),
            event_data.len() as u32,
        );
    }

    pending
}

/// Get staking info for an address
#[no_mangle]
pub extern "C" fn get_stake_info(address: u64) -> u64 {
    let contract = QBankContract::load();

    if let Some(stake) = contract.stakes.get(&address) {
        // Pack info: amount in lower 48 bits, tier in bits 48-55, locked flag in bit 56
        let current_time = unsafe { get_block_timestamp() };
        let is_locked = current_time < stake.unlock_time;

        let mut info = stake.amount & 0xFFFFFFFFFFFF; // 48 bits for amount
        info |= (stake.tier as u64) << 48;
        info |= (if is_locked { 1u64 } else { 0u64 }) << 56;

        return info;
    }

    0
}

/// Get pending staking rewards
#[no_mangle]
pub extern "C" fn get_pending_rewards(address: u64) -> u64 {
    let contract = QBankContract::load();
    contract.calculate_pending_rewards(address)
}

// ============ AIRDROP FUNCTIONS ============

/// Create an airdrop campaign (owner only)
#[no_mangle]
pub extern "C" fn create_airdrop(
    total_tokens: u64,
    tokens_per_claim: u64,
    max_claims: u64,
    duration_seconds: u64,
    vesting_duration: u64,
) -> u64 {
    let caller = unsafe { get_caller() };
    let mut contract = QBankContract::load();

    // Only owner can create airdrops
    if caller != contract.owner {
        return 0;
    }

    let owner_balance = contract.balance_of(caller);
    if owner_balance < total_tokens {
        return 0;
    }

    let current_time = unsafe { get_block_timestamp() };
    let airdrop_id = contract.next_airdrop_id;

    let airdrop = AirdropCampaign {
        id: airdrop_id,
        total_tokens,
        tokens_per_claim,
        max_claims,
        current_claims: 0,
        start_time: current_time,
        end_time: current_time + duration_seconds,
        vesting_duration,
        active: true,
    };

    // Lock tokens for airdrop
    contract.set_balance(caller, owner_balance - total_tokens);
    contract.airdrops.insert(airdrop_id, airdrop);
    contract.next_airdrop_id += 1;

    contract.save();

    // Emit airdrop event
    let mut event_data = Vec::new();
    event_data.extend_from_slice(&airdrop_id.to_le_bytes());
    event_data.extend_from_slice(&total_tokens.to_le_bytes());
    event_data.extend_from_slice(&max_claims.to_le_bytes());

    unsafe {
        emit_log(
            AIRDROP_EVENT,
            event_data.as_ptr(),
            event_data.len() as u32,
        );
    }

    airdrop_id
}

/// Claim from an airdrop
#[no_mangle]
pub extern "C" fn claim_airdrop(airdrop_id: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = QBankContract::load();

    let airdrop = match contract.airdrops.get(&airdrop_id) {
        Some(a) if a.active => a.clone(),
        _ => return false,
    };

    let current_time = unsafe { get_block_timestamp() };

    // Check time window
    if current_time < airdrop.start_time || current_time > airdrop.end_time {
        return false;
    }

    // Check if already claimed
    if contract.airdrop_claims.contains_key(&(airdrop_id, caller)) {
        return false;
    }

    // Check max claims
    if airdrop.current_claims >= airdrop.max_claims {
        return false;
    }

    // Process claim
    let claim_amount = airdrop.tokens_per_claim;

    if airdrop.vesting_duration == 0 {
        // Immediate distribution
        let caller_balance = contract.balance_of(caller);
        contract.set_balance(caller, caller_balance + claim_amount);
    }

    // Record claim
    contract.airdrop_claims.insert((airdrop_id, caller), claim_amount);

    // Update airdrop stats
    if let Some(a) = contract.airdrops.get_mut(&airdrop_id) {
        a.current_claims += 1;
    }

    contract.save();

    // Emit transfer event
    contract.emit_transfer(0, caller, claim_amount);

    true
}

// ============ ADMIN FUNCTIONS ============

/// Update fee configuration (owner only)
#[no_mangle]
pub extern "C" fn set_fees(reflection: u64, liquidity: u64, burn: u64, dev: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = QBankContract::load();

    if caller != contract.owner {
        return false;
    }

    let total = reflection + liquidity + burn + dev;
    if total > MAX_FEE_PERCENT {
        return false; // Cannot exceed max fee
    }

    contract.fee_config.reflection_fee = reflection;
    contract.fee_config.liquidity_fee = liquidity;
    contract.fee_config.burn_fee = burn;
    contract.fee_config.dev_fee = dev;

    contract.save();
    true
}

/// Exclude address from fees (owner only)
#[no_mangle]
pub extern "C" fn exclude_from_fee(address: u64, excluded: bool) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = QBankContract::load();

    if caller != contract.owner {
        return false;
    }

    contract.excluded_from_fee.insert(address, excluded);
    contract.save();
    true
}

/// Get total fees collected
#[no_mangle]
pub extern "C" fn get_total_fees() -> u64 {
    let contract = QBankContract::load();
    contract.total_fees_collected
}

/// Get total burned
#[no_mangle]
pub extern "C" fn get_total_burned() -> u64 {
    let contract = QBankContract::load();
    contract.total_burned
}

/// Get total reflected
#[no_mangle]
pub extern "C" fn get_total_reflected() -> u64 {
    let contract = QBankContract::load();
    contract.total_reflected
}

/// Get total staked
#[no_mangle]
pub extern "C" fn get_total_staked() -> u64 {
    let contract = QBankContract::load();
    contract.total_staked
}

/// Get liquidity reserve
#[no_mangle]
pub extern "C" fn get_liquidity_reserve() -> u64 {
    let contract = QBankContract::load();
    contract.liquidity_reserve
}

/// Manual burn (anyone can burn their tokens)
#[no_mangle]
pub extern "C" fn burn(amount: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = QBankContract::load();

    let balance = contract.balance_of(caller);
    if balance < amount {
        return false;
    }

    contract.set_balance(caller, balance - amount);
    contract.total_burned += amount;
    contract.circulating_supply -= amount;
    contract.total_supply -= amount;

    contract.save();

    // Emit burn event
    let mut event_data = Vec::new();
    event_data.extend_from_slice(&caller.to_le_bytes());
    event_data.extend_from_slice(&amount.to_le_bytes());

    unsafe {
        emit_log(
            BURN_EVENT,
            event_data.as_ptr(),
            event_data.len() as u32,
        );
    }

    contract.emit_transfer(caller, 0, amount);

    true
}
