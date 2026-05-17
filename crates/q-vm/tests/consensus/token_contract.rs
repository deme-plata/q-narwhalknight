// Token contract in Rust for DAGKnight VM
// This is a simplified ERC20-like token contract that would be compiled to WebAssembly

use std::collections::HashMap;

// Storage keys
const TOTAL_SUPPLY_KEY: &[u8] = b"total_supply";
const OWNER_KEY: &[u8] = b"owner";
const BALANCES_PREFIX: &[u8] = b"balance_";
const ALLOWANCES_PREFIX: &[u8] = b"allowance_";

// Event types
const TRANSFER_EVENT: u8 = 1;
const APPROVAL_EVENT: u8 = 2;

// External functions exposed by the VM
extern "C" {
    fn read_storage(key_ptr: *const u8, key_len: u32, value_ptr: *mut u8, value_len: u32) -> i32;
    fn write_storage(key_ptr: *const u8, key_len: u32, value_ptr: *const u8, value_len: u32) -> i32;
    fn emit_log(event_type: u8, data_ptr: *const u8, data_len: u32) -> i32;
    fn get_caller() -> u64;
}

// State management
struct TokenContract {
    balances: HashMap<u64, u64>,
    allowances: HashMap<(u64, u64), u64>,
    total_supply: u64,
    owner: u64,
}

impl TokenContract {
    // Load contract state from storage
    fn load() -> Self {
        let mut contract = Self {
            balances: HashMap::new(),
            allowances: HashMap::new(),
            total_supply: 0,
            owner: 0,
        };
        
        // Load total supply
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
        
        // Load owner
        unsafe {
            read_storage(
                OWNER_KEY.as_ptr(),
                OWNER_KEY.len() as u32,
                buf.as_mut_ptr(),
                buf.len() as u32,
            );
        }
        contract.owner = u64::from_le_bytes(buf);
        
        // We don't load all balances and allowances here for efficiency
        // They will be loaded on demand
        
        contract
    }
    
    // Save contract state to storage
    fn save(&self) {
        // Save total supply
        let buf = self.total_supply.to_le_bytes();
        unsafe {
            write_storage(
                TOTAL_SUPPLY_KEY.as_ptr(),
                TOTAL_SUPPLY_KEY.len() as u32,
                buf.as_ptr(),
                buf.len() as u32,
            );
        }
        
        // Save owner
        let buf = self.owner.to_le_bytes();
        unsafe {
            write_storage(
                OWNER_KEY.as_ptr(),
                OWNER_KEY.len() as u32,
                buf.as_ptr(),
                buf.len() as u32,
            );
        }
        
        // Save all modified balances
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
        
        // Save all modified allowances
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
        
        // Load from storage if not in memory
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
    
    // Get allowance
    fn allowance(&mut self, owner: u64, spender: u64) -> u64 {
        if let Some(&amount) = self.allowances.get(&(owner, spender)) {
            return amount;
        }
        
        // Load from storage if not in memory
        let key = [ALLOWANCES_PREFIX, &owner.to_le_bytes(), &spender.to_le_bytes()].concat();
        let mut buf = [0u8; 8];
        unsafe {
            read_storage(
                key.as_ptr(),
                key.len() as u32,
                buf.as_mut_ptr(),
                buf.len() as u32,
            );
        }
        
        let amount = u64::from_le_bytes(buf);
        self.allowances.insert((owner, spender), amount);
        amount
    }
    
    // Set allowance
    fn set_allowance(&mut self, owner: u64, spender: u64, amount: u64) {
        self.allowances.insert((owner, spender), amount);
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
    
    // Emit approval event
    fn emit_approval(&self, owner: u64, spender: u64, amount: u64) {
        let mut event_data = Vec::new();
        event_data.extend_from_slice(&owner.to_le_bytes());
        event_data.extend_from_slice(&spender.to_le_bytes());
        event_data.extend_from_slice(&amount.to_le_bytes());
        
        unsafe {
            emit_log(
                APPROVAL_EVENT,
                event_data.as_ptr(),
                event_data.len() as u32,
            );
        }
    }
}

// Contract methods

// Constructor
#[no_mangle]
pub extern "C" fn constructor(initial_supply: u64) -> i32 {
    let mut contract = TokenContract {
        balances: HashMap::new(),
        allowances: HashMap::new(),
        total_supply: initial_supply,
        owner: unsafe { get_caller() },
    };
    
    // Assign all tokens to the contract creator
    contract.set_balance(contract.owner, initial_supply);
    
    // Save contract state
    contract.save();
    
    // Emit transfer event from zero address
    contract.emit_transfer(0, contract.owner, initial_supply);
    
    return 1; // Success
}

// Get total supply
#[no_mangle]
pub extern "C" fn total_supply() -> u64 {
    let contract = TokenContract::load();
    contract.total_supply
}

// Get token name
#[no_mangle]
pub extern "C" fn name() -> i32 {
    // For simplicity, we hardcode the name
    let name = "DAGKnight Token";
    unsafe {
        emit_log(
            0, // Standard log event
            name.as_ptr(),
            name.len() as u32,
        );
    }
    1 // Success
}

// Get token symbol
#[no_mangle]
pub extern "C" fn symbol() -> i32 {
    // For simplicity, we hardcode the symbol
    let symbol = "DAGK";
    unsafe {
        emit_log(
            0, // Standard log event
            symbol.as_ptr(),
            symbol.len() as u32,
        );
    }
    1 // Success
}

// Get token decimals
#[no_mangle]
pub extern "C" fn decimals() -> u8 {
    18 // Standard ERC20 decimals
}

// Get balance of an address
#[no_mangle]
pub extern "C" fn balance_of(address: u64) -> u64 {
    let mut contract = TokenContract::load();
    contract.balance_of(address)
}

// Transfer tokens
#[no_mangle]
pub extern "C" fn transfer(to: u64, amount: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = TokenContract::load();
    
    // Check balance
    let from_balance = contract.balance_of(caller);
    if from_balance < amount {
        return false;
    }
    
    // Update balances
    contract.set_balance(caller, from_balance - amount);
    let to_balance = contract.balance_of(to);
    contract.set_balance(to, to_balance + amount);
    
    // Save contract state
    contract.save();
    
    // Emit transfer event
    contract.emit_transfer(caller, to, amount);
    
    true
}

// Transfer tokens from one address to another (used by approve/transferFrom pattern)
#[no_mangle]
pub extern "C" fn transfer_from(from: u64, to: u64, amount: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = TokenContract::load();
    
    // Check allowance
    let allowed = contract.allowance(from, caller);
    if allowed < amount {
        return false;
    }
    
    // Check balance
    let from_balance = contract.balance_of(from);
    if from_balance < amount {
        return false;
    }
    
    // Update balances and allowance
    contract.set_balance(from, from_balance - amount);
    let to_balance = contract.balance_of(to);
    contract.set_balance(to, to_balance + amount);
    contract.set_allowance(from, caller, allowed - amount);
    
    // Save contract state
    contract.save();
    
    // Emit transfer event
    contract.emit_transfer(from, to, amount);
    
    true
}

// Approve spender to spend tokens
#[no_mangle]
pub extern "C" fn approve(spender: u64, amount: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = TokenContract::load();
    
    // Set allowance
    contract.set_allowance(caller, spender, amount);
    
    // Save contract state
    contract.save();
    
    // Emit approval event
    contract.emit_approval(caller, spender, amount);
    
    true
}

// Get allowance
#[no_mangle]
pub extern "C" fn allowance(owner: u64, spender: u64) -> u64 {
    let mut contract = TokenContract::load();
    contract.allowance(owner, spender)
}

// Mint new tokens (only callable by contract owner)
#[no_mangle]
pub extern "C" fn mint(to: u64, amount: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = TokenContract::load();
    
    // Only owner can mint
    if caller != contract.owner {
        return false;
    }
    
    // Update balances and total supply
    let to_balance = contract.balance_of(to);
    contract.set_balance(to, to_balance + amount);
    contract.total_supply += amount;
    
    // Save contract state
    contract.save();
    
    // Emit transfer event (from zero address)
    contract.emit_transfer(0, to, amount);
    
    true
}

// Burn tokens
#[no_mangle]
pub extern "C" fn burn(amount: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = TokenContract::load();
    
    // Check balance
    let caller_balance = contract.balance_of(caller);
    if caller_balance < amount {
        return false;
    }
    
    // Update balances and total supply
    contract.set_balance(caller, caller_balance - amount);
    contract.total_supply -= amount;
    
    // Save contract state
    contract.save();
    
    // Emit transfer event (to zero address)
    contract.emit_transfer(caller, 0, amount);
    
    true
}

// Utility method to check if a contract was properly initialized
#[no_mangle]
pub extern "C" fn is_initialized() -> bool {
    let contract = TokenContract::load();
    contract.total_supply > 0 && contract.owner != 0
}

// This is a helper function for testing that performs batch transfers
// It will be useful for our performance testing
#[no_mangle]
pub extern "C" fn batch_transfer(recipients: *const u64, amounts: *const u64, count: u32) -> u32 {
    let caller = unsafe { get_caller() };
    let mut contract = TokenContract::load();
    
    let mut successful = 0;
    let recipients_slice = unsafe { std::slice::from_raw_parts(recipients, count as usize) };
    let amounts_slice = unsafe { std::slice::from_raw_parts(amounts, count as usize) };
    
    // Check if caller has enough balance for all transfers
    let mut total_amount = 0;
    for i in 0..count as usize {
        total_amount += amounts_slice[i];
    }
    
    let caller_balance = contract.balance_of(caller);
    if caller_balance < total_amount {
        return 0; // Not enough balance
    }
    
    // Perform transfers
    contract.set_balance(caller, caller_balance - total_amount);
    
    for i in 0..count as usize {
        let to = recipients_slice[i];
        let amount = amounts_slice[i];
        
        let to_balance = contract.balance_of(to);
        contract.set_balance(to, to_balance + amount);
        
        // Emit transfer event
        contract.emit_transfer(caller, to, amount);
        
        successful += 1;
    }
    
    // Save contract state
    contract.save();
    
    successful
}