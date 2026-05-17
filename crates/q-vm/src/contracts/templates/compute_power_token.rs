/// Compute Power Token - AI Processing Capability Token
/// 
/// Tokenizes AI compute processing power and participation in Hydra Computatus ecosystem
/// Enables trading of processing capability and computational resource allocation

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

// Storage keys for compute power data
const COMPUTE_TOKEN_PREFIX: &[u8] = b"compute_token_";
const PROCESSING_TASK_PREFIX: &[u8] = b"processing_task_";
const PERFORMANCE_HISTORY_PREFIX: &[u8] = b"performance_history_";
const TOTAL_COMPUTE_TOKENS_KEY: &[u8] = b"total_compute_tokens";
const NEXT_COMPUTE_TOKEN_ID_KEY: &[u8] = b"next_compute_token_id";

// Event types for compute operations
const COMPUTE_TOKEN_MINT_EVENT: u8 = 20;
const PROCESSING_TASK_ASSIGNED_EVENT: u8 = 21;
const PROCESSING_TASK_COMPLETED_EVENT: u8 = 22;
const PERFORMANCE_UPDATE_EVENT: u8 = 23;
const COMPUTE_REWARD_EVENT: u8 = 24;

// External VM functions
extern "C" {
    fn read_storage(key_ptr: *const u8, key_len: u32, value_ptr: *mut u8, value_len: u32) -> i32;
    fn write_storage(key_ptr: *const u8, key_len: u32, value_ptr: *const u8, value_len: u32) -> i32;
    fn emit_log(event_type: u8, data_ptr: *const u8, data_len: u32) -> i32;
    fn get_caller() -> u64;
    fn get_block_timestamp() -> u64;
    fn get_block_number() -> u64;
    fn quantum_random() -> u64;
}

/// Compute Power Token Data Structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComputePowerToken {
    token_id: u64,
    owner: u64,
    processing_capability_tflops: u64,           // TFLOPS processing power
    specialized_accelerators: Vec<AcceleratorType>, // Hardware acceleration types
    performance_rating: u64,                     // Historical performance score (0-100000)
    reliability_score: u64,                      // Uptime and success rate (0-100000)
    energy_efficiency_tops_per_watt: u64,        // Energy efficiency rating
    supported_models: Vec<ModelType>,            // AI model types supported
    current_tasks: Vec<ProcessingTask>,          // Active processing tasks
    completed_tasks: u64,                        // Total tasks completed
    total_inferences: u64,                       // Total AI inferences processed
    total_tokens_earned: u64,                    // QNK tokens earned from compute work
    staking_pool: u64,                          // Tokens staked for performance guarantee
    slashing_history: Vec<SlashingRecord>,       // Performance violations
    delegation_allowances: HashMap<u64, u64>,   // Delegated compute allowances
    creation_time: u64,                         // Token creation timestamp
    last_activity: u64,                         // Last processing activity
    multi_chain_bridges: Vec<CrossChainBridge>, // Cross-chain compute availability
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum AcceleratorType {
    CPU,
    GPU(String),        // GPU model/brand
    TPU,
    FPGA(String),       // FPGA configuration
    QuantumProcessor,
    NeuromorphicChip,
    AsicMiner,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ModelType {
    LLM(String),        // Large Language Model (e.g., "Llama-70B")
    DiffusionModel,     // Image generation
    VisionModel,        // Computer vision
    AudioModel,         // Speech/audio processing
    ReinforcementLearning,
    TimeSeriesForecasting,
    GraphNeuralNetwork,
    QuantumML,          // Quantum machine learning models
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProcessingTask {
    task_id: u64,
    requester: u64,
    model_type: ModelType,
    input_size_mb: u32,
    estimated_compute_hours: f64,
    reward_qnk: u64,
    deadline: u64,               // Block timestamp deadline
    assigned_time: u64,          // When task was assigned
    progress_percentage: u8,     // 0-100 completion status
    quality_metrics: QualityMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QualityMetrics {
    accuracy_score: f64,         // Model output accuracy
    latency_ms: u64,            // Processing latency
    energy_consumed_kwh: f64,    // Energy consumption
    validation_score: u64,       // Peer validation score
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SlashingRecord {
    violation_time: u64,
    violation_type: SlashingType,
    tokens_slashed: u64,
    reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum SlashingType {
    FailedTask,                 // Task not completed by deadline
    QualityViolation,          // Poor output quality
    ResourceMisrepresentation, // Fake capability claims
    EnergyWaste,               // Excessive energy consumption
    DoubleTaking,              // Taking multiple identical tasks
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CrossChainBridge {
    chain_id: u8,              // 1=Bitcoin, 2=Solana, 3=Monero, 4=Arbitrum, 5=QNK
    bridge_address: [u8; 32],  // Bridge contract address
    staked_amount: u64,        // Tokens staked on that chain
    processing_history: u64,   // Tasks completed on that chain
    reputation_score: u64,     // Cross-chain reputation
}

/// Compute Power Token Contract
struct ComputePowerContract {
    tokens: HashMap<u64, ComputePowerToken>,
    next_token_id: u64,
    total_tokens: u64,
    global_task_queue: Vec<ProcessingTask>,
    performance_leaderboard: Vec<u64>, // Token IDs sorted by performance
}

impl ComputePowerContract {
    /// Load contract state from VM storage
    fn load() -> Self {
        let mut contract = Self {
            tokens: HashMap::new(),
            next_token_id: 1,
            total_tokens: 0,
            global_task_queue: Vec::new(),
            performance_leaderboard: Vec::new(),
        };
        
        // Load next token ID
        let mut buf = [0u8; 8];
        unsafe {
            read_storage(
                NEXT_COMPUTE_TOKEN_ID_KEY.as_ptr(),
                NEXT_COMPUTE_TOKEN_ID_KEY.len() as u32,
                buf.as_mut_ptr(),
                buf.len() as u32,
            );
        }
        contract.next_token_id = u64::from_le_bytes(buf);
        if contract.next_token_id == 0 {
            contract.next_token_id = 1;
        }
        
        // Load total tokens
        unsafe {
            read_storage(
                TOTAL_COMPUTE_TOKENS_KEY.as_ptr(),
                TOTAL_COMPUTE_TOKENS_KEY.len() as u32,
                buf.as_mut_ptr(),
                buf.len() as u32,
            );
        }
        contract.total_tokens = u64::from_le_bytes(buf);
        
        contract
    }
    
    /// Save contract state to VM storage
    fn save(&self) {
        // Save next token ID
        let buf = self.next_token_id.to_le_bytes();
        unsafe {
            write_storage(
                NEXT_COMPUTE_TOKEN_ID_KEY.as_ptr(),
                NEXT_COMPUTE_TOKEN_ID_KEY.len() as u32,
                buf.as_ptr(),
                buf.len() as u32,
            );
        }
        
        // Save total tokens
        let buf = self.total_tokens.to_le_bytes();
        unsafe {
            write_storage(
                TOTAL_COMPUTE_TOKENS_KEY.as_ptr(),
                TOTAL_COMPUTE_TOKENS_KEY.len() as u32,
                buf.as_ptr(),
                buf.len() as u32,
            );
        }
        
        // Save all modified tokens
        for (token_id, token) in &self.tokens {
            self.save_compute_token(*token_id, token);
        }
    }
    
    fn save_compute_token(&self, token_id: u64, token: &ComputePowerToken) {
        let serialized = bincode::serialize(token).unwrap_or_default();
        let key = [COMPUTE_TOKEN_PREFIX, &token_id.to_le_bytes()].concat();
        
        unsafe {
            write_storage(
                key.as_ptr(),
                key.len() as u32,
                serialized.as_ptr(),
                serialized.len() as u32,
            );
        }
    }
    
    fn load_compute_token(&mut self, token_id: u64) -> Option<ComputePowerToken> {
        if let Some(token) = self.tokens.get(&token_id) {
            return Some(token.clone());
        }
        
        let key = [COMPUTE_TOKEN_PREFIX, &token_id.to_le_bytes()].concat();
        let mut buf = vec![0u8; 8192]; // Max token data size
        
        let result = unsafe {
            read_storage(
                key.as_ptr(),
                key.len() as u32,
                buf.as_mut_ptr(),
                buf.len() as u32,
            )
        };
        
        if result > 0 {
            if let Ok(token) = bincode::deserialize::<ComputePowerToken>(&buf[..result as usize]) {
                self.tokens.insert(token_id, token.clone());
                return Some(token);
            }
        }
        
        None
    }
    
    /// Calculate compute power score based on capabilities and history
    fn calculate_compute_score(&self, token: &ComputePowerToken) -> u64 {
        let capability_score = token.processing_capability_tflops * 1000;
        let performance_score = token.performance_rating;
        let reliability_score = token.reliability_score;
        let efficiency_score = token.energy_efficiency_tops_per_watt * 10;
        let accelerator_bonus = token.specialized_accelerators.len() as u64 * 5000;
        let model_support_bonus = token.supported_models.len() as u64 * 2000;
        
        // Weighted scoring with emphasis on reliability and performance
        (capability_score * 20 + performance_score * 30 + reliability_score * 40 + 
         efficiency_score * 5 + accelerator_bonus + model_support_bonus) / 100
    }
    
    /// Emit compute power event
    fn emit_compute_event(&self, event_type: u8, token: &ComputePowerToken, additional_data: &[u8]) {
        let mut event_data = Vec::new();
        event_data.extend_from_slice(&token.token_id.to_le_bytes());
        event_data.extend_from_slice(&token.owner.to_le_bytes());
        event_data.extend_from_slice(&token.processing_capability_tflops.to_le_bytes());
        event_data.extend_from_slice(additional_data);
        
        unsafe {
            emit_log(
                event_type,
                event_data.as_ptr(),
                event_data.len() as u32,
            );
        }
    }
}

// Smart contract functions for compute power token management

/// Mint new compute power token representing processing capability
#[no_mangle]
pub extern "C" fn mint_compute_power_token(
    processing_capability_tflops: u64,
    accelerator_types: u64, // Bitfield representing accelerator types
    energy_efficiency: u64,
    initial_stake: u64
) -> u64 {
    let caller = unsafe { get_caller() };
    let mut contract = ComputePowerContract::load();
    
    let token_id = contract.next_token_id;
    
    // Decode accelerator types from bitfield
    let mut accelerators = Vec::new();
    for i in 0..8 {
        if (accelerator_types >> i) & 1 == 1 {
            accelerators.push(match i {
                0 => AcceleratorType::CPU,
                1 => AcceleratorType::GPU("Generic".to_string()),
                2 => AcceleratorType::TPU,
                3 => AcceleratorType::FPGA("Generic".to_string()),
                4 => AcceleratorType::QuantumProcessor,
                5 => AcceleratorType::NeuromorphicChip,
                6 => AcceleratorType::AsicMiner,
                _ => AcceleratorType::CPU,
            });
        }
    }
    
    let compute_token = ComputePowerToken {
        token_id,
        owner: caller,
        processing_capability_tflops,
        specialized_accelerators: accelerators,
        performance_rating: 75000, // Start with good baseline performance
        reliability_score: 80000,  // Start with good reliability
        energy_efficiency_tops_per_watt: energy_efficiency,
        supported_models: vec![
            ModelType::LLM("General".to_string()),
            ModelType::VisionModel,
        ], // Default supported models
        current_tasks: Vec::new(),
        completed_tasks: 0,
        total_inferences: 0,
        total_tokens_earned: 0,
        staking_pool: initial_stake,
        slashing_history: Vec::new(),
        delegation_allowances: HashMap::new(),
        creation_time: unsafe { get_block_timestamp() },
        last_activity: unsafe { get_block_timestamp() },
        multi_chain_bridges: Vec::new(),
    };
    
    contract.tokens.insert(token_id, compute_token.clone());
    contract.next_token_id += 1;
    contract.total_tokens += 1;
    
    contract.save();
    
    // Emit mint event
    let mint_data = [
        &processing_capability_tflops.to_le_bytes(),
        &energy_efficiency.to_le_bytes(),
        &initial_stake.to_le_bytes(),
    ].concat();
    contract.emit_compute_event(COMPUTE_TOKEN_MINT_EVENT, &compute_token, &mint_data);
    
    token_id
}

/// Assign processing task to compute power token
#[no_mangle]
pub extern "C" fn assign_processing_task(
    token_id: u64,
    task_type: u8,           // ModelType encoded as u8
    input_size_mb: u32,
    estimated_hours: u32,    // Estimated compute hours * 1000 for precision
    reward_qnk: u64,
    deadline_hours: u32
) -> u64 {
    let caller = unsafe { get_caller() };
    let mut contract = ComputePowerContract::load();
    
    let mut token = match contract.load_compute_token(token_id) {
        Some(t) => t,
        None => return 0,
    };
    
    // Verify compute capability and availability
    if token.current_tasks.len() >= 5 { // Max 5 concurrent tasks
        return 0;
    }
    
    // Create new processing task
    let task_id = unsafe { quantum_random() };
    let model_type = match task_type {
        1 => ModelType::LLM("Inference".to_string()),
        2 => ModelType::DiffusionModel,
        3 => ModelType::VisionModel,
        4 => ModelType::AudioModel,
        5 => ModelType::ReinforcementLearning,
        6 => ModelType::TimeSeriesForecasting,
        7 => ModelType::GraphNeuralNetwork,
        8 => ModelType::QuantumML,
        _ => ModelType::LLM("General".to_string()),
    };
    
    let processing_task = ProcessingTask {
        task_id,
        requester: caller,
        model_type,
        input_size_mb,
        estimated_compute_hours: estimated_hours as f64 / 1000.0,
        reward_qnk,
        deadline: unsafe { get_block_timestamp() } + (deadline_hours as u64 * 3600),
        assigned_time: unsafe { get_block_timestamp() },
        progress_percentage: 0,
        quality_metrics: QualityMetrics {
            accuracy_score: 0.0,
            latency_ms: 0,
            energy_consumed_kwh: 0.0,
            validation_score: 0,
        },
    };
    
    // Reserve staking pool for task guarantee
    let stake_required = reward_qnk / 10; // 10% stake requirement
    if token.staking_pool < stake_required {
        return 0; // Insufficient stake
    }
    
    token.current_tasks.push(processing_task.clone());
    token.last_activity = unsafe { get_block_timestamp() };
    
    contract.tokens.insert(token_id, token.clone());
    contract.save();
    
    // Emit task assignment event
    let task_data = [
        &task_id.to_le_bytes(),
        &(task_type as u64).to_le_bytes(),
        &reward_qnk.to_le_bytes(),
        &deadline_hours.to_le_bytes(),
    ].concat();
    contract.emit_compute_event(PROCESSING_TASK_ASSIGNED_EVENT, &token, &task_data);
    
    task_id
}

/// Complete processing task and claim rewards
#[no_mangle]
pub extern "C" fn complete_processing_task(
    token_id: u64,
    task_id: u64,
    quality_score: u64,     // Quality metrics packed into u64
    energy_consumed: u32    // Energy consumed in Wh
) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = ComputePowerContract::load();
    
    let mut token = match contract.load_compute_token(token_id) {
        Some(t) => t,
        None => return false,
    };
    
    // Verify ownership
    if token.owner != caller {
        return false;
    }
    
    // Find and remove completed task
    let task_index = token.current_tasks.iter().position(|t| t.task_id == task_id);
    let completed_task = match task_index {
        Some(index) => token.current_tasks.remove(index),
        None => return false,
    };
    
    let current_time = unsafe { get_block_timestamp() };
    
    // Check if task was completed on time
    let on_time = current_time <= completed_task.deadline;
    let processing_time = current_time - completed_task.assigned_time;
    
    // Unpack quality metrics from u64
    let accuracy = ((quality_score >> 48) & 0xFFFF) as f64 / 1000.0; // 16 bits for accuracy
    let latency = (quality_score >> 32) & 0xFFFF;                    // 16 bits for latency
    let validation = (quality_score >> 16) & 0xFFFF;                 // 16 bits for validation
    let efficiency_bonus = quality_score & 0xFFFF;                   // 16 bits for efficiency
    
    // Calculate performance score and rewards
    let mut final_reward = completed_task.reward_qnk;
    let mut performance_impact = 0i64;
    
    if on_time && accuracy > 0.8 && validation > 8000 {
        // Excellent performance: bonus rewards
        final_reward = (final_reward as f64 * 1.2) as u64;
        performance_impact = 1000;
        token.performance_rating = std::cmp::min(token.performance_rating + 500, 100000);
        token.reliability_score = std::cmp::min(token.reliability_score + 300, 100000);
    } else if on_time && accuracy > 0.6 {
        // Good performance: standard rewards
        performance_impact = 200;
        token.performance_rating = std::cmp::min(token.performance_rating + 100, 100000);
    } else {
        // Poor performance: reduced rewards and potential slashing
        final_reward = (final_reward as f64 * 0.7) as u64;
        performance_impact = -500;
        token.performance_rating = token.performance_rating.saturating_sub(200);
        token.reliability_score = token.reliability_score.saturating_sub(300);
        
        // Record slashing if performance is very poor
        if accuracy < 0.5 || !on_time {
            let slashing_amount = completed_task.reward_qnk / 20; // 5% slashing
            token.staking_pool = token.staking_pool.saturating_sub(slashing_amount);
            
            let slashing_record = SlashingRecord {
                violation_time: current_time,
                violation_type: if !on_time { SlashingType::FailedTask } else { SlashingType::QualityViolation },
                tokens_slashed: slashing_amount,
                reason: format!("Task {} performance below threshold", task_id),
            };
            token.slashing_history.push(slashing_record);
        }
    }
    
    // Update statistics
    token.completed_tasks += 1;
    token.total_inferences += (completed_task.input_size_mb as u64 * 1000); // Estimate inferences
    token.total_tokens_earned += final_reward;
    token.last_activity = current_time;
    
    contract.tokens.insert(token_id, token.clone());
    contract.save();
    
    // Emit task completion event
    let completion_data = [
        &task_id.to_le_bytes(),
        &final_reward.to_le_bytes(),
        &(accuracy as u64).to_le_bytes(),
        &(if on_time { 1u64 } else { 0u64 }).to_le_bytes(),
    ].concat();
    contract.emit_compute_event(PROCESSING_TASK_COMPLETED_EVENT, &token, &completion_data);
    
    true
}

/// Delegate compute power to another address
#[no_mangle]
pub extern "C" fn delegate_compute_power(
    token_id: u64,
    delegate_to: u64,
    percentage: u8  // 0-100 percentage of compute power to delegate
) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = ComputePowerContract::load();
    
    let mut token = match contract.load_compute_token(token_id) {
        Some(t) => t,
        None => return false,
    };
    
    // Verify ownership
    if token.owner != caller {
        return false;
    }
    
    // Validate percentage
    if percentage > 100 {
        return false;
    }
    
    // Calculate delegated compute power
    let delegated_tflops = (token.processing_capability_tflops * percentage as u64) / 100;
    
    // Update delegation allowances
    token.delegation_allowances.insert(delegate_to, delegated_tflops);
    
    contract.tokens.insert(token_id, token.clone());
    contract.save();
    
    true
}

/// Stake additional tokens to improve reliability score
#[no_mangle]
pub extern "C" fn stake_for_reliability(token_id: u64, stake_amount: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = ComputePowerContract::load();
    
    let mut token = match contract.load_compute_token(token_id) {
        Some(t) => t,
        None => return false,
    };
    
    // Verify ownership
    if token.owner != caller {
        return false;
    }
    
    // Add to staking pool and improve reliability
    token.staking_pool += stake_amount;
    let reliability_improvement = (stake_amount / 1000).min(5000); // Max 5000 improvement per stake
    token.reliability_score = std::cmp::min(token.reliability_score + reliability_improvement, 100000);
    
    contract.tokens.insert(token_id, token.clone());
    contract.save();
    
    true
}

/// Upgrade compute token capabilities
#[no_mangle]
pub extern "C" fn upgrade_compute_capabilities(
    token_id: u64,
    new_accelerator_type: u8,
    capability_boost_tflops: u64,
    upgrade_cost: u64
) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = ComputePowerContract::load();
    
    let mut token = match contract.load_compute_token(token_id) {
        Some(t) => t,
        None => return false,
    };
    
    // Verify ownership and sufficient stake
    if token.owner != caller || token.staking_pool < upgrade_cost {
        return false;
    }
    
    // Add new accelerator
    let new_accelerator = match new_accelerator_type {
        1 => AcceleratorType::GPU("Upgraded".to_string()),
        2 => AcceleratorType::TPU,
        3 => AcceleratorType::FPGA("Custom".to_string()),
        4 => AcceleratorType::QuantumProcessor,
        5 => AcceleratorType::NeuromorphicChip,
        _ => AcceleratorType::CPU,
    };
    
    // Upgrade capabilities
    token.specialized_accelerators.push(new_accelerator);
    token.processing_capability_tflops += capability_boost_tflops;
    token.staking_pool -= upgrade_cost;
    
    // Add new model support based on accelerator type
    match new_accelerator_type {
        4 => token.supported_models.push(ModelType::QuantumML),
        5 => token.supported_models.push(ModelType::ReinforcementLearning),
        _ => {}
    }
    
    contract.tokens.insert(token_id, token.clone());
    contract.save();
    
    true
}

/// Get compute power token information
#[no_mangle]
pub extern "C" fn get_compute_token_info(token_id: u64) -> u64 {
    let mut contract = ComputePowerContract::load();
    
    if let Some(token) = contract.load_compute_token(token_id) {
        let compute_score = contract.calculate_compute_score(&token);
        
        // Pack token info into u64
        let mut info = 0u64;
        info |= (token.processing_capability_tflops & 0xFFFF);          // Capability (16 bits)
        info |= ((token.performance_rating / 1000) & 0xFF) << 16;       // Performance (8 bits)
        info |= ((token.reliability_score / 1000) & 0xFF) << 24;        // Reliability (8 bits)
        info |= (token.current_tasks.len() as u64 & 0xFF) << 32;        // Active tasks (8 bits)
        info |= ((token.completed_tasks & 0xFFFF) << 40);               // Completed tasks (16 bits)
        info |= ((compute_score / 10000) & 0xFF) << 56;                 // Overall score (8 bits)
        
        return info;
    }
    
    0
}

/// Transfer compute power token ownership
#[no_mangle]
pub extern "C" fn transfer_compute_token(token_id: u64, to: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = ComputePowerContract::load();
    
    let mut token = match contract.load_compute_token(token_id) {
        Some(t) => t,
        None => return false,
    };
    
    // Verify ownership
    if token.owner != caller {
        return false;
    }
    
    // Cannot transfer while tasks are active
    if !token.current_tasks.is_empty() {
        return false;
    }
    
    // Transfer ownership
    token.owner = to;
    token.delegation_allowances.clear(); // Clear all delegations on transfer
    
    contract.tokens.insert(token_id, token.clone());
    contract.save();
    
    true
}

/// Burn compute power token and reclaim staked tokens
#[no_mangle]
pub extern "C" fn burn_compute_token(token_id: u64) -> u64 {
    let caller = unsafe { get_caller() };
    let mut contract = ComputePowerContract::load();
    
    let token = match contract.load_compute_token(token_id) {
        Some(t) => t,
        None => return 0,
    };
    
    // Verify ownership
    if token.owner != caller {
        return 0;
    }
    
    // Cannot burn while tasks are active
    if !token.current_tasks.is_empty() {
        return 0;
    }
    
    // Return staked tokens minus any slashing penalties
    let returned_tokens = token.staking_pool;
    
    // Remove token from contract
    contract.tokens.remove(&token_id);
    contract.total_tokens -= 1;
    
    contract.save();
    
    returned_tokens
}

/// Get global compute statistics
#[no_mangle]
pub extern "C" fn get_global_compute_stats() -> u64 {
    let contract = ComputePowerContract::load();
    
    let mut total_tflops = 0u64;
    let mut active_tokens = 0u64;
    let mut total_tasks_completed = 0u64;
    let mut total_earnings = 0u64;
    
    for token in contract.tokens.values() {
        total_tflops += token.processing_capability_tflops;
        if token.last_activity > unsafe { get_block_timestamp() } - 86400 { // Active in last 24h
            active_tokens += 1;
        }
        total_tasks_completed += token.completed_tasks;
        total_earnings += token.total_tokens_earned;
    }
    
    // Pack global stats into u64
    let mut stats = 0u64;
    stats |= (total_tflops & 0xFFFF);                               // Total TFLOPS (16 bits)
    stats |= (active_tokens & 0xFF) << 16;                          // Active tokens (8 bits)
    stats |= ((total_tasks_completed & 0xFFFF) << 24);              // Total tasks (16 bits)
    stats |= ((total_earnings / 1000000) & 0xFFFFFF) << 40;         // Total earnings in millions (24 bits)
    
    stats
}

/// Create cross-chain compute bridge
#[no_mangle]
pub extern "C" fn create_cross_chain_bridge(
    token_id: u64,
    target_chain_id: u8,
    bridge_stake: u64
) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = ComputePowerContract::load();
    
    let mut token = match contract.load_compute_token(token_id) {
        Some(t) => t,
        None => return false,
    };
    
    // Verify ownership and sufficient stake
    if token.owner != caller || token.staking_pool < bridge_stake {
        return false;
    }
    
    // Generate bridge address using quantum randomness
    let mut bridge_address = [0u8; 32];
    for i in 0..4 {
        let rand_val = unsafe { quantum_random() };
        bridge_address[i*8..(i+1)*8].copy_from_slice(&rand_val.to_le_bytes());
    }
    
    let bridge = CrossChainBridge {
        chain_id: target_chain_id,
        bridge_address,
        staked_amount: bridge_stake,
        processing_history: 0,
        reputation_score: 50000, // Start with neutral reputation
    };
    
    token.multi_chain_bridges.push(bridge);
    token.staking_pool -= bridge_stake;
    
    contract.tokens.insert(token_id, token.clone());
    contract.save();
    
    true
}