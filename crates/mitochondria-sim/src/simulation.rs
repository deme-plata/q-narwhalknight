/// Mitochondria v2 Simulation Engine
///
/// Simulates the complete water-robot blockchain ecosystem including:
/// - Droplet locomotion via electro-wetting
/// - DNA blockchain synthesis and reading  
/// - Tor-controlled command & control
/// - Biological consensus via mass spectroscopy
/// - Binary fission replication

use crate::*;
use anyhow::Result;
use nalgebra::{Point2, Vector2};
use std::collections::HashMap;
use tokio::time::{interval, Duration};
use tracing::{info, debug, warn};

pub struct MitochondriaSimulation {
    pub network: MitochondriaNetwork,
    pub config: SimulationConfig,
    physics_engine: PhysicsEngine,
    dna_synthesizer: DNASynthesizer,
    tor_interface: TorInterface,
    visualization_state: VisualizationState,
}

#[derive(Debug)]
struct PhysicsEngine {
    surface_tension: f64,
    viscosity: f64,
    electro_wetting_force: f64,
    brownian_motion_coefficient: f64,
}

#[derive(Debug)]
struct DNASynthesizer {
    synthesis_rate_pg_per_ms: f64,
    error_rate: f64,
    energy_cost_per_base: f64,
    polymerase_efficiency: f64,
}

#[derive(Debug)]
struct TorInterface {
    command_latency_ms: u64,
    packet_loss_rate: f64,
    encryption_overhead: f64,
}

#[derive(Debug)]
struct VisualizationState {
    camera_position: Point2<f64>,
    zoom_level: f64,
    tracked_droplets: Vec<String>,
    swarm_colors: HashMap<String, (f32, f32, f32)>, // RGB colors for swarms
}

impl MitochondriaSimulation {
    pub async fn new(config: SimulationConfig) -> Result<Self> {
        info!("🧬 Creating Mitochondria v2 simulation");
        
        let network = create_mitochondria_simulation(config.clone()).await?;
        
        let physics_engine = PhysicsEngine {
            surface_tension: 0.072,      // Water surface tension (N/m)
            viscosity: 0.001,            // Water viscosity (Pa·s)
            electro_wetting_force: 1e-9, // Force per pad (N)
            brownian_motion_coefficient: 1e-12, // Random motion
        };
        
        let dna_synthesizer = DNASynthesizer {
            synthesis_rate_pg_per_ms: 0.0001, // Very slow DNA synthesis
            error_rate: 0.001,                // 0.1% error rate
            energy_cost_per_base: 0.01,       // Energy per DNA base
            polymerase_efficiency: 0.95,      // 95% efficient
        };
        
        let tor_interface = TorInterface {
            command_latency_ms: config.tor_command_latency_ms,
            packet_loss_rate: 0.01,  // 1% packet loss via Tor
            encryption_overhead: 0.1, // 10% overhead for encryption
        };
        
        let visualization_state = VisualizationState {
            camera_position: Point2::new(config.grid_size_mm / 2.0, config.grid_size_mm / 2.0),
            zoom_level: 1.0,
            tracked_droplets: Vec::new(),
            swarm_colors: HashMap::new(),
        };
        
        Ok(Self {
            network,
            config,
            physics_engine,
            dna_synthesizer,
            tor_interface,
            visualization_state,
        })
    }
    
    /// Run the complete simulation
    pub async fn run_simulation(&mut self, duration_seconds: u64) -> Result<SimulationReport> {
        info!("🚀 Starting Mitochondria v2 simulation for {} seconds", duration_seconds);
        
        let mut simulation_interval = interval(Duration::from_millis(
            (1000.0 / self.config.simulation_speed) as u64
        ));
        
        let start_time = std::time::Instant::now();
        let mut tick_count = 0u64;
        
        while start_time.elapsed().as_secs() < duration_seconds {
            simulation_interval.tick().await;
            
            // Simulation tick
            self.simulation_tick().await?;
            tick_count += 1;
            
            // Log progress every 1000 ticks
            if tick_count % 1000 == 0 {
                self.log_simulation_progress(tick_count).await;
            }
        }
        
        let report = self.generate_simulation_report(tick_count).await;
        info!("✅ Simulation completed: {} ticks in {} seconds", tick_count, duration_seconds);
        
        Ok(report)
    }
    
    /// Single simulation tick (physics + biology + consensus)
    async fn simulation_tick(&mut self) -> Result<()> {
        // 1. Process Tor commands
        self.process_tor_commands().await?;
        
        // 2. Update droplet physics (movement)
        self.update_droplet_physics().await?;
        
        // 3. Process DNA synthesis
        self.process_dna_synthesis().await?;
        
        // 4. Check for binary fission
        self.check_for_replication().await?;
        
        // 5. Run consensus round
        self.run_consensus_round().await?;
        
        // 6. Update visualization
        self.update_visualization().await?;
        
        // Advance simulation time
        self.network.simulation_time += chrono::Duration::milliseconds(
            (1000.0 / self.config.simulation_speed) as i64
        );
        
        Ok(())
    }
    
    /// Process commands from Tor network
    async fn process_tor_commands(&mut self) -> Result<()> {
        let commands_to_process: Vec<_> = self.network.tor_command_center.command_queue
            .iter()
            .filter(|cmd| !cmd.executed)
            .cloned()
            .collect();
        
        for mut command in commands_to_process {
            // Simulate Tor latency
            let command_age = (Utc::now() - command.issued_at).num_milliseconds() as u64;
            if command_age < self.tor_interface.command_latency_ms {
                continue; // Command still in transit
            }
            
            // Execute command
            self.execute_tor_command(&mut command).await?;
            
            // Mark as executed
            if let Some(cmd) = self.network.tor_command_center.command_queue
                .iter_mut()
                .find(|c| c.command_id == command.command_id) {
                cmd.executed = true;
            }
        }
        
        Ok(())
    }
    
    /// Execute a specific Tor command on target droplet
    async fn execute_tor_command(&mut self, command: &mut TorCommand) -> Result<()> {
        let target_droplet_id = command.target_droplet.clone();
        
        // Handle commands that need special borrowing first
        match &command.command_type {
            CommandType::EmergencyEvaporate => {
                warn!("💨 Emergency evaporation for droplet {}", target_droplet_id);
                self.evaporate_droplet(&target_droplet_id).await?;
                return Ok(());
            },
            _ => {} // Handle other commands below
        }
        
        // Pre-calculate values that need immutable self access
        let (velocity, distance_for_fret) = match &command.command_type {
            CommandType::Move { direction, distance_um } => {
                (Some(self.calculate_movement_velocity(direction, *distance_um)), None)
            },
            CommandType::ReadNeighborDNA { target_droplet_id } => {
                if let Some(droplet) = self.network.droplets.get(&command.target_droplet) {
                    if let Some(neighbor) = self.network.droplets.get(target_droplet_id) {
                        let distance = self.calculate_distance(&droplet.position, &neighbor.position);
                        (None, Some(distance))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            },
            _ => (None, None)
        };
        
        // Now get mutable access and execute commands
        let droplet = self.network.droplets.get_mut(&target_droplet_id)
            .ok_or_else(|| anyhow::anyhow!("Droplet not found: {}", target_droplet_id))?;
        
        match &command.command_type {
            CommandType::Move { direction, distance_um } => {
                if let Some(vel) = velocity {
                    droplet.position.velocity_x = vel.x;
                    droplet.position.velocity_y = vel.y;
                    debug!("🏃 Droplet {} moving {:?} {}µm", droplet.droplet_id, direction, distance_um);
                }
            },
            
            CommandType::ReadNeighborDNA { target_droplet_id } => {
                if let Some(distance) = distance_for_fret {
                    if distance < 500.0 {
                        debug!("🔬 Droplet {} reading DNA from neighbor {} (distance: {:.1}µm)", 
                               droplet.droplet_id, target_droplet_id, distance);
                        // Simplified FRET reading - just update droplet state
                        droplet.energy_level += 0.1; // Small energy gain from reading
                    }
                }
            },
            
            CommandType::SynthesizeBlock { block_data } => {
                debug!("🧬 Droplet {} synthesizing new DNA block", droplet.droplet_id);
                // Simplified synthesis - just update the droplet directly
                if droplet.energy_level > 0.3 {
                    droplet.dna_data.chain_length += 1;
                    droplet.dna_data.total_mass_picograms += block_data.len() as f64 * 0.001;
                    droplet.energy_level -= 0.2;
                }
            },
            
            CommandType::InitiateFission => {
                if droplet.size_nanoliters > 100.0 { // Simplified threshold
                    debug!("🔄 Droplet {} ready for binary fission", droplet.droplet_id);
                    droplet.replication_readiness = 1.0;
                }
            },
            
            CommandType::JoinSwarm { swarm_leader } => {
                debug!("🐝 Droplet {} joining swarm led by {}", droplet.droplet_id, swarm_leader);
                droplet.last_consensus_vote = Some(chrono::Utc::now());
            },
            
            CommandType::EmergencyEvaporate => {
                // Already handled above
                unreachable!()
            },
            
            CommandType::BuildCircuit => {
                debug!("🔧 Droplet {} building Tor circuit", droplet.droplet_id);
                droplet.tor_connection_id = format!("circuit_build_{}", chrono::Utc::now().timestamp());
            },
            
            CommandType::AssignCircuit => {
                debug!("📡 Droplet {} assigned to Tor circuit", droplet.droplet_id);
                droplet.tor_connection_id = format!("circuit_assigned_{}", droplet.droplet_id);
            },
            
            CommandType::SendMessage => {
                debug!("📤 Droplet {} sending message through Tor", droplet.droplet_id);
                droplet.energy_level -= 0.05; // Small energy cost for message
            },
            
            CommandType::UpdateRoute => {
                debug!("🗺️ Droplet {} updating Tor route", droplet.droplet_id);
                droplet.tor_connection_id = format!("route_update_{}", chrono::Utc::now().timestamp());
            },
        }
        
        Ok(())
    }
    
    /// Update physics for all droplets
    async fn update_droplet_physics(&mut self) -> Result<()> {
        for droplet in self.network.droplets.values_mut() {
            // Apply electro-wetting locomotion
            droplet.position.x += droplet.position.velocity_x * (1.0 / self.config.simulation_speed);
            droplet.position.y += droplet.position.velocity_y * (1.0 / self.config.simulation_speed);
            
            // Apply friction and surface tension
            droplet.position.velocity_x *= 0.95; // Friction
            droplet.position.velocity_y *= 0.95;
            
            // Add Brownian motion (quantum noise)
            let brownian_x = (rand::random::<f64>() - 0.5) * self.config.quantum_noise_level;
            let brownian_y = (rand::random::<f64>() - 0.5) * self.config.quantum_noise_level;
            
            droplet.position.x += brownian_x;
            droplet.position.y += brownian_y;
            
            // Boundary conditions (keep droplets on grid)
            droplet.position.x = droplet.position.x.max(0.0).min(self.config.grid_size_mm);
            droplet.position.y = droplet.position.y.max(0.0).min(self.config.grid_size_mm);
            
            // Energy decay
            droplet.energy_level -= self.config.energy_decay_rate / self.config.simulation_speed;
            droplet.energy_level = droplet.energy_level.max(0.0);
        }
        
        Ok(())
    }
    
    /// Process DNA synthesis for active droplets
    async fn process_dna_synthesis(&mut self) -> Result<()> {
        for droplet in self.network.droplets.values_mut() {
            if droplet.energy_level > 0.1 { // Need minimum energy for synthesis
                // Synthesize DNA based on available energy
                let synthesis_amount = self.config.dna_synthesis_rate / self.config.simulation_speed;
                droplet.dna_data.total_mass_picograms += synthesis_amount;
                
                // Energy cost for synthesis
                droplet.energy_level -= synthesis_amount * 0.1;
            }
        }
        
        Ok(())
    }
    
    /// Check droplets for binary fission readiness
    async fn check_for_replication(&mut self) -> Result<()> {
        let droplets_to_replicate: Vec<_> = self.network.droplets.values()
            .filter(|d| d.size_nanoliters > self.config.fission_threshold)
            .map(|d| d.droplet_id.clone())
            .collect();
        
        for droplet_id in droplets_to_replicate {
            self.perform_binary_fission(&droplet_id).await?;
        }
        
        Ok(())
    }
    
    /// Perform binary fission on a droplet
    async fn perform_binary_fission(&mut self, parent_id: &str) -> Result<()> {
        let parent = self.network.droplets.get(parent_id)
            .ok_or_else(|| anyhow::anyhow!("Parent droplet not found"))?
            .clone();
        
        info!("🔄 Binary fission: {} → two daughters", parent_id);
        
        // Create two daughter droplets
        let daughter1_id = format!("{}_d1", parent_id);
        let daughter2_id = format!("{}_d2", parent_id);
        
        let mut daughter1 = parent.clone();
        daughter1.droplet_id = daughter1_id.clone();
        daughter1.size_nanoliters = parent.size_nanoliters / 2.0;
        daughter1.dna_data.total_mass_picograms = parent.dna_data.total_mass_picograms / 2.0;
        daughter1.position.x += 0.1; // Slight position offset
        
        let mut daughter2 = parent.clone();
        daughter2.droplet_id = daughter2_id.clone();
        daughter2.size_nanoliters = parent.size_nanoliters / 2.0;
        daughter2.dna_data.total_mass_picograms = parent.dna_data.total_mass_picograms / 2.0;
        daughter2.position.x -= 0.1; // Slight position offset
        
        // Add daughters to network
        self.network.droplets.insert(daughter1_id, daughter1);
        self.network.droplets.insert(daughter2_id, daughter2);
        
        // Remove parent (it has divided)
        self.network.droplets.remove(parent_id);
        
        info!("✅ Binary fission completed: {} daughters created", 2);
        Ok(())
    }
    
    /// Run biological consensus round
    async fn run_consensus_round(&mut self) -> Result<()> {
        // Update total network DNA mass
        self.network.consensus_state.total_network_dna_mass = crate::dna_storage::calculate_total_dna_mass(&self.network.droplets);
        
        // Find new consensus leader (heaviest DNA mass)
        let new_leader = crate::dna_storage::find_heaviest_droplet(&self.network.droplets);
        
        if new_leader != self.network.consensus_state.heaviest_swarm_leader {
            info!("👑 New consensus leader: {} (mass: {:.2} pg)", 
                  new_leader, 
                  self.network.droplets.get(&new_leader)
                      .map(|d| d.dna_data.total_mass_picograms)
                      .unwrap_or(0.0));
            
            self.network.consensus_state.heaviest_swarm_leader = new_leader;
            self.network.consensus_state.last_consensus_round = Utc::now();
        }
        
        // Calculate consensus confidence based on mass distribution
        self.network.consensus_state.consensus_confidence = self.calculate_consensus_confidence();
        
        Ok(())
    }
    
    /// Calculate consensus confidence based on DNA mass distribution
    fn calculate_consensus_confidence(&self) -> f64 {
        if self.network.droplets.is_empty() {
            return 0.0;
        }
        
        let total_mass = self.network.consensus_state.total_network_dna_mass;
        if total_mass == 0.0 {
            return 0.0;
        }
        
        // Find leader's mass
        let leader_mass = self.network.droplets.get(&self.network.consensus_state.heaviest_swarm_leader)
            .map(|d| d.dna_data.total_mass_picograms)
            .unwrap_or(0.0);
        
        // Confidence = leader's percentage of total mass
        leader_mass / total_mass
    }
    
    /// Calculate movement velocity from direction and distance
    fn calculate_movement_velocity(&self, direction: &Direction, distance_um: f64) -> Vector2<f64> {
        let speed = distance_um / 1000.0; // Convert µm to mm
        
        match direction {
            Direction::North => Vector2::new(0.0, speed),
            Direction::South => Vector2::new(0.0, -speed),
            Direction::East => Vector2::new(speed, 0.0),
            Direction::West => Vector2::new(-speed, 0.0),
            Direction::Northeast => Vector2::new(speed * 0.707, speed * 0.707),
            Direction::Northwest => Vector2::new(-speed * 0.707, speed * 0.707),
            Direction::Southeast => Vector2::new(speed * 0.707, -speed * 0.707),
            Direction::Southwest => Vector2::new(-speed * 0.707, -speed * 0.707),
        }
    }
    
    /// Calculate distance between two droplets
    fn calculate_distance(&self, pos1: &Position2D, pos2: &Position2D) -> f64 {
        let dx = pos1.x - pos2.x;
        let dy = pos1.y - pos2.y;
        (dx * dx + dy * dy).sqrt()
    }
    
    /// Simulate FRET fluorescence reading between droplets
    async fn simulate_fret_reading(&self, reader: &DropletNode, target: &DropletNode) -> Result<()> {
        debug!("🔬 FRET reading: {} scanning {}", reader.droplet_id, target.droplet_id);
        
        // Simulate fluorescence detection of DNA sequences
        let detection_efficiency = 0.9; // 90% detection rate
        let signal_strength = target.dna_data.total_mass_picograms * detection_efficiency;
        
        if signal_strength > 0.1 { // Minimum detectable signal
            debug!("✅ FRET reading successful: detected {:.2} pg DNA", signal_strength);
        } else {
            debug!("❌ FRET reading failed: signal too weak");
        }
        
        Ok(())
    }
    
    /// Synthesize new DNA block in droplet
    async fn synthesize_dna_block(&self, droplet: &mut DropletNode, block_data: &[u8]) -> Result<()> {
        if droplet.energy_level < 0.5 {
            return Err(anyhow::anyhow!("Insufficient energy for DNA synthesis"));
        }
        
        // Convert block data to DNA sequence
        let dna_sequence = self.encode_block_data_to_dna(block_data);
        
        // Calculate synthesis cost
        let synthesis_cost = dna_sequence.len() as f64 * self.dna_synthesizer.energy_cost_per_base;
        
        if droplet.energy_level < synthesis_cost {
            return Err(anyhow::anyhow!("Insufficient energy for this block size"));
        }
        
        // Perform synthesis
        let synthesis_time = (dna_sequence.len() as f64 / self.dna_synthesizer.synthesis_rate_pg_per_ms) as u64;
        
        droplet.dna_data.synthesis_history.push(DNASynthesisEvent {
            block_height: droplet.dna_data.chain_length as u64,
            sequence_added: dna_sequence.clone(),
            synthesis_time_ms: synthesis_time,
            energy_cost: synthesis_cost,
            synthesized_at: Utc::now(),
        });
        
        // Update DNA blockchain
        droplet.dna_data.chain_length += 1;
        droplet.dna_data.latest_block_hash = hex::encode(blake3::hash(block_data).as_bytes());
        droplet.dna_data.total_mass_picograms += dna_sequence.len() as f64 * 0.001; // ~1 fg per base
        
        // Consume energy
        droplet.energy_level -= synthesis_cost;
        
        debug!("🧬 DNA synthesis completed: {} bases, {:.2} pg added", 
               dna_sequence.len(), dna_sequence.len() as f64 * 0.001);
        
        Ok(())
    }
    
    /// Encode block data into DNA sequence
    fn encode_block_data_to_dna(&self, block_data: &[u8]) -> String {
        let mut dna_sequence = String::new();
        
        for byte in block_data {
            // Map each byte to DNA bases (2 bits per base)
            let base1 = match (byte >> 6) & 0x03 {
                0 => "A",
                1 => "T", 
                2 => "G",
                3 => "C",
                _ => unreachable!(),
            };
            let base2 = match (byte >> 4) & 0x03 {
                0 => "A",
                1 => "T",
                2 => "G", 
                3 => "C",
                _ => unreachable!(),
            };
            let base3 = match (byte >> 2) & 0x03 {
                0 => "A",
                1 => "T",
                2 => "G",
                3 => "C", 
                _ => unreachable!(),
            };
            let base4 = match byte & 0x03 {
                0 => "A",
                1 => "T",
                2 => "G",
                3 => "C",
                _ => unreachable!(),
            };
            
            dna_sequence.push_str(&format!("{}{}{}{}", base1, base2, base3, base4));
        }
        
        dna_sequence
    }
    
    /// Initiate droplet fission
    async fn initiate_droplet_fission(&self, droplet: &mut DropletNode) -> Result<()> {
        droplet.replication_readiness = 1.0;
        info!("🔄 Droplet {} ready for fission", droplet.droplet_id);
        Ok(())
    }
    
    /// Join droplet to swarm
    async fn join_droplet_swarm(&self, droplet: &mut DropletNode, _swarm_leader: &str) -> Result<()> {
        debug!("🐝 Droplet {} joining swarm", droplet.droplet_id);
        // Move towards swarm leader (simplified)
        droplet.position.velocity_x *= 1.1;
        droplet.position.velocity_y *= 1.1;
        Ok(())
    }
    
    /// Evaporate droplet (removal from simulation)
    async fn evaporate_droplet(&mut self, droplet_id: &str) -> Result<()> {
        self.network.droplets.remove(droplet_id);
        info!("💨 Droplet {} evaporated", droplet_id);
        Ok(())
    }
    
    /// Update visualization state
    async fn update_visualization(&mut self) -> Result<()> {
        // Update tracked droplets (follow most interesting ones)
        self.visualization_state.tracked_droplets = self.network.droplets.keys()
            .take(10)
            .cloned()
            .collect();
        
        // Update swarm colors based on DNA similarity
        for droplet in self.network.droplets.values() {
            let color_hash = blake3::hash(droplet.dna_data.latest_block_hash.as_bytes());
            let color = (
                color_hash.as_bytes()[0] as f32 / 255.0,
                color_hash.as_bytes()[1] as f32 / 255.0,
                color_hash.as_bytes()[2] as f32 / 255.0,
            );
            self.visualization_state.swarm_colors.insert(droplet.droplet_id.clone(), color);
        }
        
        Ok(())
    }
    
    /// Log simulation progress
    async fn log_simulation_progress(&self, tick_count: u64) {
        let droplet_count = self.network.droplets.len();
        let total_mass = self.network.consensus_state.total_network_dna_mass;
        let leader = &self.network.consensus_state.heaviest_swarm_leader;
        
        info!("📊 Simulation tick {}: {} droplets, {:.2} pg total DNA, leader: {}", 
              tick_count, droplet_count, total_mass, leader);
    }
    
    /// Generate comprehensive simulation report
    async fn generate_simulation_report(&self, total_ticks: u64) -> SimulationReport {
        let droplet_count = self.network.droplets.len();
        let total_dna_mass = self.network.consensus_state.total_network_dna_mass;
        let consensus_rounds = total_ticks / 100; // Consensus every 100 ticks
        
        let average_droplet_size = self.network.droplets.values()
            .map(|d| d.size_nanoliters)
            .sum::<f64>() / droplet_count as f64;
        
        let total_synthesis_events = self.network.droplets.values()
            .map(|d| d.dna_data.synthesis_history.len())
            .sum::<usize>();
        
        SimulationReport {
            simulation_duration_ticks: total_ticks,
            final_droplet_count: droplet_count,
            total_dna_mass_picograms: total_dna_mass,
            consensus_rounds_completed: consensus_rounds,
            binary_fissions_occurred: self.count_fission_events(),
            average_droplet_size_nl: average_droplet_size,
            total_synthesis_events,
            consensus_leader: self.network.consensus_state.heaviest_swarm_leader.clone(),
            final_consensus_confidence: self.network.consensus_state.consensus_confidence,
            tor_commands_processed: self.network.tor_command_center.command_queue.len(),
        }
    }
    
    fn count_fission_events(&self) -> usize {
        self.network.droplets.keys()
            .filter(|id| id.contains("_d1") || id.contains("_d2"))
            .count() / 2 // Each fission creates 2 daughters
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulationReport {
    pub simulation_duration_ticks: u64,
    pub final_droplet_count: usize,
    pub total_dna_mass_picograms: f64,
    pub consensus_rounds_completed: u64,
    pub binary_fissions_occurred: usize,
    pub average_droplet_size_nl: f64,
    pub total_synthesis_events: usize,
    pub consensus_leader: String,
    pub final_consensus_confidence: f64,
    pub tor_commands_processed: usize,
}

/// Tor command generation for external control
pub async fn generate_tor_command_from_bitcoin_header() -> Result<TorCommand> {
    info!("🧅 Generating Tor command from Bitcoin header");
    
    // Get latest Bitcoin block (simplified)
    let latest_block_hash = hex::encode(rand::random::<[u8; 32]>());
    
    // Generate 4-byte command from block hash
    let hash = blake3::hash(latest_block_hash.as_bytes());
    let command_bytes = &hash.as_bytes()[..4];
    
    // Decode command type from bytes
    let command_type = match command_bytes[0] % 6 {
        0 => CommandType::Move { 
            direction: Direction::North, 
            distance_um: (command_bytes[1] as f64) * 10.0 
        },
        1 => CommandType::ReadNeighborDNA { 
            target_droplet_id: format!("droplet_{:04}", command_bytes[1] as usize) 
        },
        2 => CommandType::SynthesizeBlock { 
            block_data: command_bytes.to_vec() 
        },
        3 => CommandType::InitiateFission,
        4 => CommandType::JoinSwarm { 
            swarm_leader: format!("droplet_{:04}", command_bytes[2] as usize) 
        },
        5 => CommandType::EmergencyEvaporate,
        _ => unreachable!(),
    };
    
    Ok(TorCommand {
        command_id: uuid::Uuid::new_v4().to_string(),
        target_droplet: format!("droplet_{:04}", command_bytes[3] as usize),
        command_type,
        payload: command_bytes.to_vec(),
        issued_at: Utc::now(),
        executed: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_simulation_creation() {
        let config = SimulationConfig::default();
        let sim = MitochondriaSimulation::new(config).await.unwrap();
        
        assert!(!sim.network.droplets.is_empty());
        assert_eq!(sim.network.droplets.len(), 100);
    }
    
    #[test]
    fn test_dna_encoding() {
        let sim = MitochondriaSimulation::new(SimulationConfig::default()).await.unwrap();
        let block_data = b"test block";
        let dna_sequence = sim.encode_block_data_to_dna(block_data);
        
        assert_eq!(dna_sequence.len(), block_data.len() * 4); // 4 bases per byte
        assert!(dna_sequence.chars().all(|c| matches!(c, 'A' | 'T' | 'G' | 'C')));
    }
    
    #[test]
    fn test_consensus_confidence() {
        let mut sim = MitochondriaSimulation::new(SimulationConfig::default()).await.unwrap();
        
        // Set up test scenario with clear leader
        if let Some(leader) = sim.network.droplets.values_mut().next() {
            leader.dna_data.total_mass_picograms = 100.0; // Much larger than others
        }
        
        let confidence = sim.calculate_consensus_confidence();
        assert!(confidence > 0.5); // Should have high confidence with clear leader
    }
}