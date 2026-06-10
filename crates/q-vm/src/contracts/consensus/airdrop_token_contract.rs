// Enhanced Token Contract with Airdrop Functionality for DAGKnight VM
// Extends the basic ERC20-like token contract with comprehensive airdrop capabilities

use std::collections::HashMap;

// Storage keys
const TOTAL_SUPPLY_KEY: &[u8] = b"total_supply";
const OWNER_KEY: &[u8] = b"owner";
const BALANCES_PREFIX: &[u8] = b"balance_";
const ALLOWANCES_PREFIX: &[u8] = b"allowance_";
const AIRDROP_PREFIX: &[u8] = b"airdrop_";
const CLAIM_PREFIX: &[u8] = b"claim_";
const ELIGIBILITY_PREFIX: &[u8] = b"eligible_";

// Event types
const TRANSFER_EVENT: u8 = 1;
const APPROVAL_EVENT: u8 = 2;
const AIRDROP_CREATED_EVENT: u8 = 3;
const AIRDROP_CLAIMED_EVENT: u8 = 4;
const MINT_EVENT: u8 = 5;
const BURN_EVENT: u8 = 6;

// External functions exposed by the VM
extern "C" {
    fn read_storage(key_ptr: *const u8, key_len: u32, value_ptr: *mut u8, value_len: u32) -> i32;
    fn write_storage(key_ptr: *const u8, key_len: u32, value_ptr: *const u8, value_len: u32) -> i32;
    fn emit_log(event_type: u8, data_ptr: *const u8, data_len: u32) -> i32;
    fn get_caller() -> u64;
    fn get_block_timestamp() -> u64;
    fn get_block_number() -> u64;
}

// Enhanced contract state with airdrop functionality
struct AirdropTokenContract {
    balances: HashMap<u64, u64>,
    allowances: HashMap<(u64, u64), u64>,
    total_supply: u64,
    owner: u64,
    airdrops: HashMap<u64, AirdropInfo>,
    claims: HashMap<(u64, u64), ClaimInfo>, // (airdrop_id, address) -> claim_info
    eligibility: HashMap<(u64, u64), bool>, // (airdrop_id, address) -> eligible
    next_airdrop_id: u64,
}

#[derive(Clone)]
struct AirdropInfo {
    id: u64,
    creator: u64,
    total_tokens: u64,
    tokens_per_participant: u64,
    max_participants: u64,
    current_participants: u64,
    claimed_tokens: u64,
    start_time: u64,
    end_time: u64,
    eligibility_type: u8, // 0 = Whitelist, 1 = Snapshot, 2 = Open
    vesting_enabled: bool,
    vesting_duration: u64,
    cliff_period: u64,
    active: bool,
}

#[derive(Clone)]
struct ClaimInfo {
    claimed_amount: u64,
    claim_time: u64,
    vested_amount: u64,
    last_vest_claim: u64,
    total_vesting: u64,
}

impl AirdropTokenContract {
    // Load contract state from storage
    fn load() -> Self {
        let mut contract = Self {
            balances: HashMap::new(),
            allowances: HashMap::new(),
            total_supply: 0,
            owner: 0,
            airdrops: HashMap::new(),
            claims: HashMap::new(),
            eligibility: HashMap::new(),
            next_airdrop_id: 1,
        };
        
        // Load basic token data
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
        
        contract
    }
    
    // Save contract state to storage
    fn save(&self) {
        // Save basic token data
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
    
    // Get balance for an address
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
    
    // Set balance for an address
    fn set_balance(&mut self, address: u64, balance: u64) {
        self.balances.insert(address, balance);
    }
    
    // Emit transfer event
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
    
    // Emit airdrop created event
    fn emit_airdrop_created(&self, airdrop_id: u64, creator: u64, total_tokens: u64) {
        let mut event_data = Vec::new();
        event_data.extend_from_slice(&airdrop_id.to_le_bytes());
        event_data.extend_from_slice(&creator.to_le_bytes());
        event_data.extend_from_slice(&total_tokens.to_le_bytes());
        
        unsafe {
            emit_log(
                AIRDROP_CREATED_EVENT,
                event_data.as_ptr(),
                event_data.len() as u32,
            );
        }
    }
    
    // Emit airdrop claimed event
    fn emit_airdrop_claimed(&self, airdrop_id: u64, claimant: u64, amount: u64) {
        let mut event_data = Vec::new();
        event_data.extend_from_slice(&airdrop_id.to_le_bytes());
        event_data.extend_from_slice(&claimant.to_le_bytes());
        event_data.extend_from_slice(&amount.to_le_bytes());
        
        unsafe {
            emit_log(
                AIRDROP_CLAIMED_EVENT,
                event_data.as_ptr(),
                event_data.len() as u32,
            );
        }
    }
    
    // Check if address is eligible for airdrop
    fn is_eligible(&self, airdrop_id: u64, address: u64) -> bool {
        self.eligibility.get(&(airdrop_id, address)).unwrap_or(&false).clone()
    }
    
    // Set eligibility for address
    fn set_eligibility(&mut self, airdrop_id: u64, address: u64, eligible: bool) {
        self.eligibility.insert((airdrop_id, address), eligible);
    }
    
    // Get available vested tokens for claim
    fn get_available_vested_tokens(&self, airdrop_id: u64, address: u64) -> u64 {
        if let Some(airdrop) = self.airdrops.get(&airdrop_id) {
            if let Some(claim) = self.claims.get(&(airdrop_id, address)) {
                if !airdrop.vesting_enabled {
                    return 0; // No vesting
                }
                
                let current_time = unsafe { get_block_timestamp() };
                
                // Check if cliff period has passed
                if current_time < claim.claim_time + airdrop.cliff_period {
                    return 0;
                }
                
                // Calculate vested amount
                let time_since_claim = current_time - claim.claim_time;
                let vesting_progress = if time_since_claim >= airdrop.vesting_duration {
                    1.0
                } else {
                    time_since_claim as f64 / airdrop.vesting_duration as f64
                };
                
                let total_vested = (claim.total_vesting as f64 * vesting_progress) as u64;
                return total_vested.saturating_sub(claim.vested_amount);
            }
        }
        0
    }
}

// Basic token functions (from original contract)

#[no_mangle]
pub extern "C" fn constructor(initial_supply: u64) -> i32 {
    let mut contract = AirdropTokenContract {
        balances: HashMap::new(),
        allowances: HashMap::new(),
        total_supply: initial_supply,
        owner: unsafe { get_caller() },
        airdrops: HashMap::new(),
        claims: HashMap::new(),
        eligibility: HashMap::new(),
        next_airdrop_id: 1,
    };
    
    contract.set_balance(contract.owner, initial_supply);
    contract.save();
    contract.emit_transfer(0, contract.owner, initial_supply);
    
    1 // Success
}

#[no_mangle]
pub extern "C" fn total_supply() -> u64 {
    let contract = AirdropTokenContract::load();
    contract.total_supply
}

#[no_mangle]
pub extern "C" fn balance_of(address: u64) -> u64 {
    let mut contract = AirdropTokenContract::load();
    contract.balance_of(address)
}

#[no_mangle]
pub extern "C" fn transfer(to: u64, amount: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = AirdropTokenContract::load();
    
    let from_balance = contract.balance_of(caller);
    if from_balance < amount {
        return false;
    }
    
    contract.set_balance(caller, from_balance - amount);
    let to_balance = contract.balance_of(to);
    contract.set_balance(to, to_balance + amount);
    
    contract.save();
    contract.emit_transfer(caller, to, amount);
    
    true
}

// Enhanced airdrop functions

#[no_mangle]
pub extern "C" fn create_airdrop(
    total_tokens: u64,
    tokens_per_participant: u64,
    max_participants: u64,
    duration: u64,
    eligibility_type: u8,
    vesting_enabled: bool,
    vesting_duration: u64,
    cliff_period: u64,
) -> u64 {
    let caller = unsafe { get_caller() };
    let mut contract = AirdropTokenContract::load();
    
    // Only owner can create airdrops
    if caller != contract.owner {
        return 0; // Failure
    }
    
    // Check if creator has enough tokens
    let creator_balance = contract.balance_of(caller);
    if creator_balance < total_tokens {
        return 0; // Insufficient balance
    }
    
    let current_time = unsafe { get_block_timestamp() };
    let airdrop_id = contract.next_airdrop_id;
    
    let airdrop = AirdropInfo {
        id: airdrop_id,
        creator: caller,
        total_tokens,
        tokens_per_participant,
        max_participants,
        current_participants: 0,
        claimed_tokens: 0,
        start_time: current_time,
        end_time: current_time + duration,
        eligibility_type,
        vesting_enabled,
        vesting_duration,
        cliff_period,
        active: true,
    };
    
    // Lock tokens for airdrop (transfer to contract)
    contract.set_balance(caller, creator_balance - total_tokens);
    
    contract.airdrops.insert(airdrop_id, airdrop);
    contract.next_airdrop_id += 1;
    
    contract.save();
    contract.emit_airdrop_created(airdrop_id, caller, total_tokens);
    
    airdrop_id
}

#[no_mangle]
pub extern "C" fn set_airdrop_eligibility(airdrop_id: u64, address: u64, eligible: bool) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = AirdropTokenContract::load();
    
    // Check if airdrop exists and caller is authorized
    if let Some(airdrop) = contract.airdrops.get(&airdrop_id) {
        if caller != airdrop.creator && caller != contract.owner {
            return false; // Unauthorized
        }
        
        contract.set_eligibility(airdrop_id, address, eligible);
        contract.save();
        return true;
    }
    
    false
}

#[no_mangle]
pub extern "C" fn batch_set_eligibility(
    airdrop_id: u64,
    addresses: *const u64,
    eligible_flags: *const bool,
    count: u32,
) -> u32 {
    let caller = unsafe { get_caller() };
    let mut contract = AirdropTokenContract::load();
    
    // Check authorization
    if let Some(airdrop) = contract.airdrops.get(&airdrop_id) {
        if caller != airdrop.creator && caller != contract.owner {
            return 0; // Unauthorized
        }
    } else {
        return 0; // Airdrop not found
    }
    
    let addresses_slice = unsafe { std::slice::from_raw_parts(addresses, count as usize) };
    let eligible_slice = unsafe { std::slice::from_raw_parts(eligible_flags, count as usize) };
    
    let mut successful = 0;
    for i in 0..count as usize {
        contract.set_eligibility(airdrop_id, addresses_slice[i], eligible_slice[i]);
        successful += 1;
    }
    
    contract.save();
    successful
}

#[no_mangle]
pub extern "C" fn claim_airdrop(airdrop_id: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = AirdropTokenContract::load();
    
    // Check if airdrop exists and is active
    let airdrop = match contract.airdrops.get(&airdrop_id) {
        Some(airdrop) if airdrop.active => airdrop.clone(),
        _ => return false,
    };
    
    let current_time = unsafe { get_block_timestamp() };
    
    // Check if airdrop is within time window
    if current_time < airdrop.start_time || current_time > airdrop.end_time {
        return false;
    }
    
    // Check eligibility
    if !contract.is_eligible(airdrop_id, caller) {
        return false;
    }
    
    // Check if already claimed
    if contract.claims.contains_key(&(airdrop_id, caller)) {
        return false;
    }
    
    // Check if max participants reached
    if airdrop.current_participants >= airdrop.max_participants {
        return false;
    }
    
    let claim_amount = airdrop.tokens_per_participant;
    
    // Create claim record
    let claim_info = ClaimInfo {
        claimed_amount: if airdrop.vesting_enabled { 0 } else { claim_amount },
        claim_time: current_time,
        vested_amount: 0,
        last_vest_claim: current_time,
        total_vesting: if airdrop.vesting_enabled { claim_amount } else { 0 },
    };
    
    contract.claims.insert((airdrop_id, caller), claim_info);
    
    // If no vesting, transfer tokens immediately
    if !airdrop.vesting_enabled {
        let caller_balance = contract.balance_of(caller);
        contract.set_balance(caller, caller_balance + claim_amount);
    }
    
    // Update airdrop statistics
    if let Some(airdrop_mut) = contract.airdrops.get_mut(&airdrop_id) {
        airdrop_mut.current_participants += 1;
        airdrop_mut.claimed_tokens += claim_amount;
    }
    
    contract.save();
    contract.emit_airdrop_claimed(airdrop_id, caller, claim_amount);
    
    true
}

#[no_mangle]
pub extern "C" fn claim_vested_tokens(airdrop_id: u64) -> u64 {
    let caller = unsafe { get_caller() };
    let mut contract = AirdropTokenContract::load();
    
    let available_tokens = contract.get_available_vested_tokens(airdrop_id, caller);
    if available_tokens == 0 {
        return 0;
    }
    
    // Update claim record
    if let Some(claim) = contract.claims.get_mut(&(airdrop_id, caller)) {
        claim.vested_amount += available_tokens;
        claim.last_vest_claim = unsafe { get_block_timestamp() };
    }
    
    // Transfer vested tokens
    let caller_balance = contract.balance_of(caller);
    contract.set_balance(caller, caller_balance + available_tokens);
    
    contract.save();
    contract.emit_transfer(0, caller, available_tokens); // From vesting contract
    
    available_tokens
}

#[no_mangle]
pub extern "C" fn get_airdrop_info(airdrop_id: u64) -> u64 {
    let contract = AirdropTokenContract::load();
    
    if let Some(airdrop) = contract.airdrops.get(&airdrop_id) {
        // Return packed airdrop info (simplified)
        // In a real implementation, this would be more sophisticated
        let mut info = 0u64;
        info |= airdrop.total_tokens & 0xFFFFFFFF;
        info |= (airdrop.current_participants & 0xFFFF) << 32;
        info |= (if airdrop.active { 1u64 } else { 0u64 }) << 48;
        info |= (if airdrop.vesting_enabled { 1u64 } else { 0u64 }) << 49;
        
        return info;
    }
    
    0
}

#[no_mangle]
pub extern "C" fn get_claim_info(airdrop_id: u64, address: u64) -> u64 {
    let contract = AirdropTokenContract::load();
    
    if let Some(claim) = contract.claims.get(&(airdrop_id, address)) {
        // Return packed claim info (simplified)
        let mut info = 0u64;
        info |= claim.claimed_amount & 0xFFFFFFFF;
        info |= (claim.vested_amount & 0xFFFFFFFF) << 32;
        
        return info;
    }
    
    0
}

#[no_mangle]
pub extern "C" fn get_available_vested_tokens(airdrop_id: u64, address: u64) -> u64 {
    let contract = AirdropTokenContract::load();
    contract.get_available_vested_tokens(airdrop_id, address)
}

#[no_mangle]
pub extern "C" fn check_eligibility(airdrop_id: u64, address: u64) -> bool {
    let contract = AirdropTokenContract::load();
    contract.is_eligible(airdrop_id, address)
}

// Token control functions for enhanced management

#[no_mangle]
pub extern "C" fn mint_tokens(to: u64, amount: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = AirdropTokenContract::load();
    
    // Only owner can mint
    if caller != contract.owner {
        return false;
    }
    
    let to_balance = contract.balance_of(to);
    contract.set_balance(to, to_balance + amount);
    contract.total_supply += amount;
    
    contract.save();
    
    // Emit mint event
    let mut event_data = Vec::new();
    event_data.extend_from_slice(&to.to_le_bytes());
    event_data.extend_from_slice(&amount.to_le_bytes());
    
    unsafe {
        emit_log(
            MINT_EVENT,
            event_data.as_ptr(),
            event_data.len() as u32,
        );
    }
    
    contract.emit_transfer(0, to, amount);
    
    true
}

#[no_mangle]
pub extern "C" fn burn_tokens(amount: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = AirdropTokenContract::load();
    
    let caller_balance = contract.balance_of(caller);
    if caller_balance < amount {
        return false;
    }
    
    contract.set_balance(caller, caller_balance - amount);
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

#[no_mangle]
pub extern "C" fn emergency_pause_airdrop(airdrop_id: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = AirdropTokenContract::load();
    
    // Only owner can emergency pause
    if caller != contract.owner {
        return false;
    }
    
    if let Some(airdrop) = contract.airdrops.get_mut(&airdrop_id) {
        airdrop.active = false;
        contract.save();
        return true;
    }
    
    false
}

#[no_mangle]
pub extern "C" fn resume_airdrop(airdrop_id: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = AirdropTokenContract::load();
    
    // Only owner can resume
    if caller != contract.owner {
        return false;
    }
    
    if let Some(airdrop) = contract.airdrops.get_mut(&airdrop_id) {
        airdrop.active = true;
        contract.save();
        return true;
    }
    
    false
}

// Batch processing for high-performance distributions
#[no_mangle]
pub extern "C" fn batch_airdrop_direct(
    recipients: *const u64,
    amounts: *const u64,
    count: u32,
) -> u32 {
    let caller = unsafe { get_caller() };
    let mut contract = AirdropTokenContract::load();
    
    // Only owner can do direct batch airdrops
    if caller != contract.owner {
        return 0;
    }
    
    let recipients_slice = unsafe { std::slice::from_raw_parts(recipients, count as usize) };
    let amounts_slice = unsafe { std::slice::from_raw_parts(amounts, count as usize) };
    
    // Calculate total amount needed
    let mut total_amount = 0;
    for i in 0..count as usize {
        total_amount += amounts_slice[i];
    }
    
    let caller_balance = contract.balance_of(caller);
    if caller_balance < total_amount {
        return 0; // Not enough balance
    }
    
    // Perform batch transfers
    contract.set_balance(caller, caller_balance - total_amount);
    
    let mut successful = 0;
    for i in 0..count as usize {
        let to = recipients_slice[i];
        let amount = amounts_slice[i];
        
        let to_balance = contract.balance_of(to);
        contract.set_balance(to, to_balance + amount);
        
        contract.emit_transfer(caller, to, amount);
        successful += 1;
    }
    
    contract.save();
    successful
}