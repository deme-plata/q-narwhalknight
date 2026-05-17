/// Hydra Organism Token - Living NFTs for Cryptobia Kingdom
/// 
/// Advanced biological computing token that represents living water-robot organisms
/// Integrates with Hydra Computatus distributed AI and multi-chain ecosystem
/// Features evolution, reproduction, metabolism, and compute contribution tracking

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

// Storage keys for organism data
const ORGANISM_PREFIX: &[u8] = b"organism_";
const GENOME_PREFIX: &[u8] = b"genome_";
const PARENT_PREFIX: &[u8] = b"parent_";
const CHILD_PREFIX: &[u8] = b"child_";
const EVOLUTION_PREFIX: &[u8] = b"evolution_";
const COMPUTE_PREFIX: &[u8] = b"compute_";
const METABOLISM_PREFIX: &[u8] = b"metabolism_";
const OWNER_TOKENS_PREFIX: &[u8] = b"owner_tokens_";
const TOTAL_ORGANISMS_KEY: &[u8] = b"total_organisms";
const NEXT_TOKEN_ID_KEY: &[u8] = b"next_token_id";

// Event types for organism lifecycle
const ORGANISM_BIRTH_EVENT: u8 = 10;
const ORGANISM_EVOLUTION_EVENT: u8 = 11;
const ORGANISM_REPRODUCTION_EVENT: u8 = 12;
const ORGANISM_DEATH_EVENT: u8 = 13;
const COMPUTE_CONTRIBUTION_EVENT: u8 = 14;
const METABOLISM_EVENT: u8 = 15;

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

/// Core organism data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OrganismData {
    token_id: u64,
    owner: u64,
    genome_hash: [u8; 32],                    // SHA-3 genetic code
    generation: u32,                          // Generation number from genesis
    birth_time: u64,                          // Block timestamp of birth
    last_metabolism: u64,                     // Last feeding/energy consumption
    energy_level: u64,                        // Current energy reserves (0-100000)
    compute_power_tflops: u64,                // AI processing capability
    specialized_accelerators: Vec<u8>,        // Accelerator types (1=GPU, 2=Quantum, etc)
    parents: (Option<u64>, Option<u64>),      // Parent organism token IDs
    children: Vec<u64>,                       // Child organism token IDs
    evolution_history: Vec<EvolutionRecord>,  // Mutation and selection history
    multi_chain_identities: Vec<ChainIdentity>, // Cross-chain presence
    survival_score: u64,                      // Cosmic survival fitness
    compute_contributions: ComputeStats,      // AI castle participation stats
    reproduction_count: u32,                  // Number of offspring produced
    death_time: Option<u64>,                  // Time of death (if deceased)
    death_cause: Option<u8>,                  // Cause of death (1=energy, 2=age, 3=conflict)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvolutionRecord {
    evolution_time: u64,
    mutation_type: u8,        // 1=beneficial, 2=neutral, 3=harmful
    fitness_delta: i32,       // Change in fitness score
    catalyst: u8,             // 1=natural, 2=radiation, 3=engineered
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChainIdentity {
    chain_id: u8,             // 1=Bitcoin, 2=Solana, 3=Monero, 4=Arbitrum, 5=QNK
    address: [u8; 32],        // Blockchain address for this organism
    balance: u64,             // Token balance on this chain
    last_transaction: u64,    // Last transaction timestamp
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComputeStats {
    total_inferences: u64,     // AI inferences processed
    tokens_earned: u64,        // QNK tokens earned from compute
    castle_participations: u32, // Number of compute castles joined
    average_performance: u64,   // Average tokens/second processing
    reliability_score: u64,     // Uptime and quality metrics
}

/// Enhanced organism contract with biological operations
struct HydraOrganismContract {
    organisms: HashMap<u64, OrganismData>,
    next_token_id: u64,
    total_organisms: u64,
    species_registry: HashMap<u8, SpeciesInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpeciesInfo {
    species_id: u8,
    species_name: String,
    base_compute_power: u64,
    base_energy_efficiency: u64,
    mutation_rate: u32,
    reproduction_cost: u64,
    specialization: u8,       // 1=compute, 2=coordination, 3=bridge, 4=storage
}

impl HydraOrganismContract {
    /// Load contract state from VM storage
    fn load() -> Self {
        let mut contract = Self {
            organisms: HashMap::new(),
            next_token_id: 1,
            total_organisms: 0,
            species_registry: HashMap::new(),
        };
        
        // Load next token ID
        let mut buf = [0u8; 8];
        unsafe {
            read_storage(
                NEXT_TOKEN_ID_KEY.as_ptr(),
                NEXT_TOKEN_ID_KEY.len() as u32,
                buf.as_mut_ptr(),
                buf.len() as u32,
            );
        }
        contract.next_token_id = u64::from_le_bytes(buf);
        if contract.next_token_id == 0 {
            contract.next_token_id = 1; // Default start
        }
        
        // Load total organisms
        unsafe {
            read_storage(
                TOTAL_ORGANISMS_KEY.as_ptr(),
                TOTAL_ORGANISMS_KEY.len() as u32,
                buf.as_mut_ptr(),
                buf.len() as u32,
            );
        }
        contract.total_organisms = u64::from_le_bytes(buf);
        
        // Initialize species registry
        contract.initialize_species_registry();
        
        contract
    }
    
    /// Save contract state to VM storage
    fn save(&self) {
        // Save next token ID
        let buf = self.next_token_id.to_le_bytes();
        unsafe {
            write_storage(
                NEXT_TOKEN_ID_KEY.as_ptr(),
                NEXT_TOKEN_ID_KEY.len() as u32,
                buf.as_ptr(),
                buf.len() as u32,
            );
        }
        
        // Save total organisms
        let buf = self.total_organisms.to_le_bytes();
        unsafe {
            write_storage(
                TOTAL_ORGANISMS_KEY.as_ptr(),
                TOTAL_ORGANISMS_KEY.len() as u32,
                buf.as_ptr(),
                buf.len() as u32,
            );
        }
        
        // Save all modified organisms
        for (token_id, organism) in &self.organisms {
            self.save_organism(*token_id, organism);
        }
    }
    
    fn save_organism(&self, token_id: u64, organism: &OrganismData) {
        // Serialize organism data
        let serialized = bincode::serialize(organism).unwrap_or_default();
        let key = [ORGANISM_PREFIX, &token_id.to_le_bytes()].concat();
        
        unsafe {
            write_storage(
                key.as_ptr(),
                key.len() as u32,
                serialized.as_ptr(),
                serialized.len() as u32,
            );
        }
    }
    
    fn load_organism(&mut self, token_id: u64) -> Option<OrganismData> {
        if let Some(organism) = self.organisms.get(&token_id) {
            return Some(organism.clone());
        }
        
        // Load from storage
        let key = [ORGANISM_PREFIX, &token_id.to_le_bytes()].concat();
        let mut buf = vec![0u8; 4096]; // Max organism data size
        
        let result = unsafe {
            read_storage(
                key.as_ptr(),
                key.len() as u32,
                buf.as_mut_ptr(),
                buf.len() as u32,
            )
        };
        
        if result > 0 {
            if let Ok(organism) = bincode::deserialize::<OrganismData>(&buf[..result as usize]) {
                self.organisms.insert(token_id, organism.clone());
                return Some(organism);
            }
        }
        
        None
    }
    
    /// Initialize species registry with Cryptobia Kingdom species
    fn initialize_species_registry(&mut self) {
        // Hydra Computatus - AI Processing Species
        self.species_registry.insert(1, SpeciesInfo {
            species_id: 1,
            species_name: "Hydra Computatus".to_string(),
            base_compute_power: 1000,      // 1 TFLOP base
            base_energy_efficiency: 500,   // 500 TOPS/W
            mutation_rate: 100,            // 1% mutation rate
            reproduction_cost: 50000,      // 50K energy to reproduce
            specialization: 1,             // Compute specialization
        });
        
        // Hydra Coordinatus - Coordination Species
        self.species_registry.insert(2, SpeciesInfo {
            species_id: 2,
            species_name: "Hydra Coordinatus".to_string(),
            base_compute_power: 500,
            base_energy_efficiency: 800,
            mutation_rate: 50,
            reproduction_cost: 30000,
            specialization: 2,             // Coordination specialization
        });
        
        // Hydra Bridgeus - Cross-chain Bridge Species
        self.species_registry.insert(3, SpeciesInfo {
            species_id: 3,
            species_name: "Hydra Bridgeus".to_string(),
            base_compute_power: 750,
            base_energy_efficiency: 600,
            mutation_rate: 75,
            reproduction_cost: 40000,
            specialization: 3,             // Bridge specialization
        });
    }
    
    /// Generate genetic code using quantum randomness
    fn generate_genetic_code(&self, parent1: Option<&OrganismData>, parent2: Option<&OrganismData>) -> [u8; 32] {
        let mut genome = [0u8; 32];
        
        match (parent1, parent2) {
            (Some(p1), Some(p2)) => {
                // Sexual reproduction: combine parent genes with mutation
                for i in 0..32 {
                    let rand_val = unsafe { quantum_random() };
                    
                    // Inherit from parent 1 or 2 with 50% probability
                    let base_gene = if rand_val % 2 == 0 {
                        p1.genome_hash[i]
                    } else {
                        p2.genome_hash[i]
                    };
                    
                    // Apply mutation with species-specific rate
                    let mutation_chance = 1000; // 0.1% base mutation rate
                    if rand_val % mutation_chance == 0 {
                        genome[i] = ((base_gene as u64 + rand_val) % 256) as u8;
                    } else {
                        genome[i] = base_gene;
                    }
                }
            },
            (Some(parent), None) => {
                // Asexual reproduction: copy with higher mutation rate
                for i in 0..32 {
                    let rand_val = unsafe { quantum_random() };
                    let mutation_chance = 500; // 0.2% mutation rate for asexual
                    
                    if rand_val % mutation_chance == 0 {
                        genome[i] = ((parent.genome_hash[i] as u64 + rand_val) % 256) as u8;
                    } else {
                        genome[i] = parent.genome_hash[i];
                    }
                }
            },
            _ => {
                // Genesis organism: pure quantum randomness
                for i in 0..32 {
                    genome[i] = (unsafe { quantum_random() } % 256) as u8;
                }
            }
        }
        
        genome
    }
    
    /// Calculate organism fitness based on survival metrics
    fn calculate_fitness(&self, organism: &OrganismData) -> u64 {
        let age_factor = (unsafe { get_block_timestamp() } - organism.birth_time) / 1000; // Age in minutes
        let energy_factor = organism.energy_level;
        let compute_factor = organism.compute_contributions.reliability_score;
        let survival_factor = organism.survival_score;
        
        // Fitness = weighted average of key survival factors
        (age_factor * 10 + energy_factor * 20 + compute_factor * 30 + survival_factor * 40) / 100
    }
    
    /// Emit organism lifecycle event
    fn emit_organism_event(&self, event_type: u8, organism: &OrganismData, additional_data: &[u8]) {
        let mut event_data = Vec::new();
        event_data.extend_from_slice(&organism.token_id.to_le_bytes());
        event_data.extend_from_slice(&organism.owner.to_le_bytes());
        event_data.extend_from_slice(&organism.genome_hash);
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

// Smart contract functions for organism management

/// Create genesis organism with quantum-generated genetics
#[no_mangle]
pub extern "C" fn create_genesis_organism(species_id: u8, initial_energy: u64) -> u64 {
    let caller = unsafe { get_caller() };
    let mut contract = HydraOrganismContract::load();
    
    let token_id = contract.next_token_id;
    let genome = contract.generate_genetic_code(None, None);
    
    // Get species info for capabilities
    let species = contract.species_registry.get(&species_id).cloned()
        .unwrap_or_else(|| contract.species_registry.get(&1).unwrap().clone()); // Default to Computatus
    
    let organism = OrganismData {
        token_id,
        owner: caller,
        genome_hash: genome,
        generation: 0,
        birth_time: unsafe { get_block_timestamp() },
        last_metabolism: unsafe { get_block_timestamp() },
        energy_level: initial_energy,
        compute_power_tflops: species.base_compute_power,
        specialized_accelerators: vec![species_id], // Species determines base accelerator
        parents: (None, None),
        children: Vec::new(),
        evolution_history: Vec::new(),
        multi_chain_identities: Vec::new(),
        survival_score: 50000, // Base survival score
        compute_contributions: ComputeStats {
            total_inferences: 0,
            tokens_earned: 0,
            castle_participations: 0,
            average_performance: 0,
            reliability_score: 80000, // Base reliability
        },
        reproduction_count: 0,
        death_time: None,
        death_cause: None,
    };
    
    contract.organisms.insert(token_id, organism.clone());
    contract.next_token_id += 1;
    contract.total_organisms += 1;
    
    contract.save();
    
    // Emit birth event
    let birth_data = [species_id, (initial_energy & 0xFF) as u8].concat();
    contract.emit_organism_event(ORGANISM_BIRTH_EVENT, &organism, &birth_data);
    
    token_id
}

/// Sexual reproduction between two organisms
#[no_mangle]
pub extern "C" fn reproduce_organisms(parent1_id: u64, parent2_id: u64, energy_cost: u64) -> u64 {
    let caller = unsafe { get_caller() };
    let mut contract = HydraOrganismContract::load();
    
    // Load parent organisms
    let parent1 = match contract.load_organism(parent1_id) {
        Some(org) => org,
        None => return 0, // Parent 1 not found
    };
    
    let parent2 = match contract.load_organism(parent2_id) {
        Some(org) => org,
        None => return 0, // Parent 2 not found
    };
    
    // Verify ownership and energy requirements
    if parent1.owner != caller || parent2.owner != caller {
        return 0; // Not authorized
    }
    
    if parent1.energy_level < energy_cost || parent2.energy_level < energy_cost {
        return 0; // Insufficient energy
    }
    
    // Check compatibility (same species or compatible species)
    let species1 = parent1.specialized_accelerators.get(0).unwrap_or(&1);
    let species2 = parent2.specialized_accelerators.get(0).unwrap_or(&1);
    if !are_species_compatible(*species1, *species2) {
        return 0; // Incompatible species
    }
    
    // Generate offspring genetics
    let child_genome = contract.generate_genetic_code(Some(&parent1), Some(&parent2));
    let generation = std::cmp::max(parent1.generation, parent2.generation) + 1;
    
    // Create child organism
    let child_token_id = contract.next_token_id;
    let child_organism = OrganismData {
        token_id: child_token_id,
        owner: caller,
        genome_hash: child_genome,
        generation,
        birth_time: unsafe { get_block_timestamp() },
        last_metabolism: unsafe { get_block_timestamp() },
        energy_level: (parent1.energy_level + parent2.energy_level) / 4, // Inherit some energy
        compute_power_tflops: (parent1.compute_power_tflops + parent2.compute_power_tflops) / 2,
        specialized_accelerators: inherit_accelerators(&parent1, &parent2),
        parents: (Some(parent1_id), Some(parent2_id)),
        children: Vec::new(),
        evolution_history: Vec::new(),
        multi_chain_identities: Vec::new(),
        survival_score: (parent1.survival_score + parent2.survival_score) / 2,
        compute_contributions: ComputeStats {
            total_inferences: 0,
            tokens_earned: 0,
            castle_participations: 0,
            average_performance: 0,
            reliability_score: 75000, // Start with good reliability
        },
        reproduction_count: 0,
        death_time: None,
        death_cause: None,
    };
    
    // Update parent energy and reproduction counts
    let mut updated_parent1 = parent1;
    let mut updated_parent2 = parent2;
    updated_parent1.energy_level -= energy_cost;
    updated_parent2.energy_level -= energy_cost;
    updated_parent1.reproduction_count += 1;
    updated_parent2.reproduction_count += 1;
    updated_parent1.children.push(child_token_id);
    updated_parent2.children.push(child_token_id);
    
    // Save all organisms
    contract.organisms.insert(parent1_id, updated_parent1);
    contract.organisms.insert(parent2_id, updated_parent2);
    contract.organisms.insert(child_token_id, child_organism.clone());
    contract.next_token_id += 1;
    contract.total_organisms += 1;
    
    contract.save();
    
    // Emit reproduction event
    let reproduction_data = [
        &parent1_id.to_le_bytes(),
        &parent2_id.to_le_bytes(),
        &generation.to_le_bytes(),
    ].concat();
    contract.emit_organism_event(ORGANISM_REPRODUCTION_EVENT, &child_organism, &reproduction_data);
    
    child_token_id
}

/// Feed organism to increase energy level
#[no_mangle]
pub extern "C" fn feed_organism(token_id: u64, energy_amount: u64, energy_type: u8) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = HydraOrganismContract::load();
    
    let mut organism = match contract.load_organism(token_id) {
        Some(org) => org,
        None => return false,
    };
    
    // Verify ownership
    if organism.owner != caller {
        return false;
    }
    
    // Check if organism is alive
    if organism.death_time.is_some() {
        return false;
    }
    
    // Apply energy with efficiency based on type
    let efficiency_multiplier = match energy_type {
        1 => 1.0,   // Standard electricity
        2 => 1.2,   // Solar light
        3 => 1.5,   // Quantum energy
        4 => 0.8,   // Waste energy
        _ => 1.0,
    };
    
    let effective_energy = (energy_amount as f64 * efficiency_multiplier) as u64;
    organism.energy_level = std::cmp::min(organism.energy_level + effective_energy, 100000);
    organism.last_metabolism = unsafe { get_block_timestamp() };
    
    contract.organisms.insert(token_id, organism.clone());
    contract.save();
    
    // Emit metabolism event
    let metabolism_data = [energy_type, (effective_energy & 0xFF) as u8].concat();
    contract.emit_organism_event(METABOLISM_EVENT, &organism, &metabolism_data);
    
    true
}

/// Apply evolutionary pressure and trigger organism evolution
#[no_mangle]
pub extern "C" fn evolve_organism(token_id: u64, pressure_type: u8, pressure_intensity: u32) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = HydraOrganismContract::load();
    
    let mut organism = match contract.load_organism(token_id) {
        Some(org) => org,
        None => return false,
    };
    
    // Verify ownership
    if organism.owner != caller {
        return false;
    }
    
    // Check if organism is alive and has enough energy for evolution
    if organism.death_time.is_some() || organism.energy_level < 20000 {
        return false;
    }
    
    // Apply evolutionary pressure
    let evolution_success = unsafe { quantum_random() } % 100 < pressure_intensity as u64;
    
    if evolution_success {
        // Successful evolution: improve capabilities
        let improvement_factor = 1.0 + (pressure_intensity as f64 / 1000.0);
        organism.compute_power_tflops = (organism.compute_power_tflops as f64 * improvement_factor) as u64;
        organism.survival_score += pressure_intensity as u64 * 10;
        
        // Add evolution record
        let evolution_record = EvolutionRecord {
            evolution_time: unsafe { get_block_timestamp() },
            mutation_type: 1, // Beneficial
            fitness_delta: pressure_intensity as i32,
            catalyst: pressure_type,
        };
        organism.evolution_history.push(evolution_record);
        
        // Consume energy for evolution
        organism.energy_level -= 20000;
        
    } else {
        // Failed evolution: neutral or harmful mutation
        let harm_factor = 0.95 + (unsafe { quantum_random() } % 10) as f64 / 100.0;
        organism.compute_power_tflops = (organism.compute_power_tflops as f64 * harm_factor) as u64;
        
        let evolution_record = EvolutionRecord {
            evolution_time: unsafe { get_block_timestamp() },
            mutation_type: if harm_factor < 1.0 { 3 } else { 2 }, // Harmful or neutral
            fitness_delta: -(pressure_intensity as i32 / 2),
            catalyst: pressure_type,
        };
        organism.evolution_history.push(evolution_record);
        
        organism.energy_level -= 10000; // Less energy consumed for failed evolution
    }
    
    contract.organisms.insert(token_id, organism.clone());
    contract.save();
    
    // Emit evolution event
    let evolution_data = [
        pressure_type,
        if evolution_success { 1 } else { 0 },
        (pressure_intensity & 0xFF) as u8,
    ].concat();
    contract.emit_organism_event(ORGANISM_EVOLUTION_EVENT, &organism, &evolution_data);
    
    evolution_success
}

/// Record AI compute contribution for organism
#[no_mangle]
pub extern "C" fn record_compute_contribution(
    token_id: u64, 
    inferences: u32, 
    tokens_earned: u64, 
    performance_score: u64
) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = HydraOrganismContract::load();
    
    let mut organism = match contract.load_organism(token_id) {
        Some(org) => org,
        None => return false,
    };
    
    // Only organism owner or compute castle can record contributions
    if organism.owner != caller {
        return false; // TODO: Add castle authorization check
    }
    
    // Update compute statistics
    organism.compute_contributions.total_inferences += inferences as u64;
    organism.compute_contributions.tokens_earned += tokens_earned;
    organism.compute_contributions.castle_participations += 1;
    
    // Update performance average
    let total_contributions = organism.compute_contributions.castle_participations as u64;
    organism.compute_contributions.average_performance = 
        (organism.compute_contributions.average_performance * (total_contributions - 1) + performance_score) 
        / total_contributions;
    
    // Increase survival score based on compute contribution
    organism.survival_score += tokens_earned / 1000; // 1 survival point per 1000 tokens earned
    
    // Add energy from successful compute work
    organism.energy_level = std::cmp::min(
        organism.energy_level + tokens_earned / 100, // 1 energy per 100 tokens
        100000
    );
    
    contract.organisms.insert(token_id, organism.clone());
    contract.save();
    
    // Emit compute contribution event
    let contribution_data = [
        &inferences.to_le_bytes(),
        &tokens_earned.to_le_bytes(),
        &performance_score.to_le_bytes(),
    ].concat();
    contract.emit_organism_event(COMPUTE_CONTRIBUTION_EVENT, &organism, &contribution_data);
    
    true
}

/// Get organism information and stats
#[no_mangle]
pub extern "C" fn get_organism_info(token_id: u64) -> u64 {
    let mut contract = HydraOrganismContract::load();
    
    if let Some(organism) = contract.load_organism(token_id) {
        // Pack organism info into u64 (simplified for demo)
        let mut info = 0u64;
        info |= organism.energy_level & 0xFFFF;                    // Energy (16 bits)
        info |= (organism.generation as u64 & 0xFF) << 16;         // Generation (8 bits)
        info |= (organism.survival_score & 0xFFFF) << 24;          // Survival (16 bits)
        info |= (organism.compute_contributions.castle_participations as u64 & 0xFF) << 40; // Participations (8 bits)
        info |= (if organism.death_time.is_some() { 1u64 } else { 0u64 }) << 48; // Alive flag (1 bit)
        
        return info;
    }
    
    0 // Organism not found
}

/// Transfer organism ownership (NFT transfer)
#[no_mangle]
pub extern "C" fn transfer_organism(token_id: u64, to: u64) -> bool {
    let caller = unsafe { get_caller() };
    let mut contract = HydraOrganismContract::load();
    
    let mut organism = match contract.load_organism(token_id) {
        Some(org) => org,
        None => return false,
    };
    
    // Verify ownership
    if organism.owner != caller {
        return false;
    }
    
    // Check if organism is alive
    if organism.death_time.is_some() {
        return false;
    }
    
    // Transfer ownership
    organism.owner = to;
    
    contract.organisms.insert(token_id, organism.clone());
    contract.save();
    
    // Emit transfer event (similar to ERC721)
    let transfer_data = [&caller.to_le_bytes(), &to.to_le_bytes()].concat();
    contract.emit_organism_event(ORGANISM_BIRTH_EVENT, &organism, &transfer_data); // Reuse birth event type
    
    true
}

/// Natural selection: Kill weak organisms and reward strong ones
#[no_mangle]
pub extern "C" fn natural_selection(selection_pressure: u32) -> u32 {
    let mut contract = HydraOrganismContract::load();
    let current_time = unsafe { get_block_timestamp() };
    
    let mut organisms_affected = 0;
    let organism_ids: Vec<u64> = contract.organisms.keys().cloned().collect();
    
    for token_id in organism_ids {
        if let Some(mut organism) = contract.load_organism(token_id) {
            let fitness = contract.calculate_fitness(&organism);
            let age_in_hours = (current_time - organism.birth_time) / 3600;
            
            // Check survival based on fitness and selection pressure
            let survival_threshold = (selection_pressure as u64 * age_in_hours) / 100;
            
            if fitness < survival_threshold && organism.death_time.is_none() {
                // Organism dies from selection pressure
                organism.death_time = Some(current_time);
                organism.death_cause = Some(3); // Death by selection
                
                contract.organisms.insert(token_id, organism.clone());
                
                // Emit death event
                let death_data = [3u8, (selection_pressure & 0xFF) as u8].concat(); // Cause: selection
                contract.emit_organism_event(ORGANISM_DEATH_EVENT, &organism, &death_data);
                
                organisms_affected += 1;
            } else if fitness > survival_threshold * 2 {
                // Strong organism gets energy bonus
                organism.energy_level = std::cmp::min(organism.energy_level + 5000, 100000);
                organism.survival_score += 100;
                
                contract.organisms.insert(token_id, organism);
                organisms_affected += 1;
            }
        }
    }
    
    contract.save();
    organisms_affected
}

/// Get ecosystem population statistics
#[no_mangle]
pub extern "C" fn get_ecosystem_stats() -> u64 {
    let contract = HydraOrganismContract::load();
    
    let mut living_organisms = 0u64;
    let mut total_compute_power = 0u64;
    let mut average_generation = 0u64;
    let mut total_energy = 0u64;
    
    for organism in contract.organisms.values() {
        if organism.death_time.is_none() {
            living_organisms += 1;
            total_compute_power += organism.compute_power_tflops;
            average_generation += organism.generation as u64;
            total_energy += organism.energy_level;
        }
    }
    
    if living_organisms > 0 {
        average_generation /= living_organisms;
    }
    
    // Pack ecosystem stats into u64
    let mut stats = 0u64;
    stats |= living_organisms & 0xFFFF;                    // Living count (16 bits)
    stats |= (total_compute_power & 0xFFFF) << 16;         // Compute power (16 bits)
    stats |= (average_generation & 0xFF) << 32;            // Avg generation (8 bits)
    stats |= (total_energy & 0xFFFFFF) << 40;              // Total energy (24 bits)
    
    stats
}

// Helper functions

fn are_species_compatible(species1: u8, species2: u8) -> bool {
    // Define compatibility matrix for species
    match (species1, species2) {
        (1, 1) => true,  // Computatus x Computatus
        (1, 2) => true,  // Computatus x Coordinatus
        (2, 2) => true,  // Coordinatus x Coordinatus
        (2, 3) => true,  // Coordinatus x Bridgeus
        (3, 3) => true,  // Bridgeus x Bridgeus
        _ => false,      // Other combinations not compatible
    }
}

fn inherit_accelerators(parent1: &OrganismData, parent2: &OrganismData) -> Vec<u8> {
    let mut accelerators = Vec::new();
    
    // Inherit from both parents with some randomness
    let mut combined = parent1.specialized_accelerators.clone();
    combined.extend(&parent2.specialized_accelerators);
    
    // Remove duplicates and apply genetic lottery
    let mut unique_accelerators: Vec<u8> = combined.into_iter().collect::<std::collections::HashSet<_>>().into_iter().collect();
    unique_accelerators.sort();
    
    // Random inheritance - each accelerator has 70% chance to be inherited
    for &acc in &unique_accelerators {
        if unsafe { quantum_random() } % 100 < 70 {
            accelerators.push(acc);
        }
    }
    
    // Ensure at least one accelerator
    if accelerators.is_empty() && !unique_accelerators.is_empty() {
        accelerators.push(unique_accelerators[0]);
    }
    
    accelerators
}

/// Get owner's organism count
#[no_mangle]
pub extern "C" fn balance_of(owner: u64) -> u64 {
    let contract = HydraOrganismContract::load();
    
    contract.organisms.values()
        .filter(|org| org.owner == owner && org.death_time.is_none())
        .count() as u64
}

/// Get organism owner
#[no_mangle]
pub extern "C" fn owner_of(token_id: u64) -> u64 {
    let mut contract = HydraOrganismContract::load();
    
    if let Some(organism) = contract.load_organism(token_id) {
        organism.owner
    } else {
        0 // Not found
    }
}

/// Check if organism exists and is alive
#[no_mangle]
pub extern "C" fn is_organism_alive(token_id: u64) -> bool {
    let mut contract = HydraOrganismContract::load();
    
    if let Some(organism) = contract.load_organism(token_id) {
        organism.death_time.is_none()
    } else {
        false
    }
}