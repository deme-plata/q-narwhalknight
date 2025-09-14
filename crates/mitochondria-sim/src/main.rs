/// Project Mitochondria v2 - Water-Robot Blockchain Simulation
///
/// "The living water-robot that mines biology instead of electricity."
///
/// This executable demonstrates the complete Mitochondria v2 ecosystem:
/// - DNA-programmed water droplets as blockchain nodes
/// - Electro-wetting locomotion system
/// - Tor-controlled command & control
/// - Biological consensus via mass spectroscopy
/// - Real-time visualization of the water-robot swarm

use mitochondria_sim::*;
use mitochondria_sim::droplet::create_genesis_droplet;
use anyhow::Result;
use tokio::io::{self, AsyncBufReadExt, BufReader};
use tracing::{info, error};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    info!("🧬💧 Project Mitochondria v2 - Water-Robot Blockchain Simulation");
    info!("🤖 Initializing biological computing system...");
    
    // Create simulation configuration
    let mut config = SimulationConfig::default();
    
    // Interactive configuration
    println!("🔧 Simulation Configuration:");
    println!("Press Enter for defaults, or type new values:");
    
    config = configure_simulation_interactively(config).await?;
    
    // Create and start simulation
    let mut simulation = simulation::MitochondriaSimulation::new(config.clone()).await?;
    
    println!("🚀 Starting Mitochondria v2 simulation...");
    println!("Commands: 'status', 'spawn', 'tor <command>', 'quit'");
    
    // Start simulation in background
    let simulation_handle = {
        let mut sim_clone = simulation.clone();
        tokio::spawn(async move {
            if let Err(e) = sim_clone.run_simulation(3600).await { // 1 hour simulation
                error!("Simulation failed: {}", e);
            }
        })
    };
    
    // Interactive command loop
    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();
    
    loop {
        print!("mitochondria> ");
        line.clear();
        
        if reader.read_line(&mut line).await? == 0 {
            break; // EOF
        }
        
        let command = line.trim();
        
        match command {
            "quit" | "exit" => {
                println!("🧬 Terminating water-robot simulation...");
                break;
            },
            
            "status" => {
                display_simulation_status(&simulation).await;
            },
            
            "spawn" => {
                spawn_new_droplet(&mut simulation).await?;
            },
            
            cmd if cmd.starts_with("tor ") => {
                let tor_cmd = &cmd[4..];
                execute_tor_command(&mut simulation, tor_cmd).await?;
            },
            
            "demo" => {
                run_demo_sequence(&mut simulation).await?;
            },
            
            "stats" => {
                display_detailed_stats(&simulation).await;
            },
            
            "help" => {
                display_help();
            },
            
            _ => {
                println!("❓ Unknown command: {}", command);
                println!("Type 'help' for available commands");
            }
        }
    }
    
    // Cleanup
    simulation_handle.abort();
    println!("✅ Mitochondria v2 simulation terminated");
    
    Ok(())
}

async fn configure_simulation_interactively(mut config: SimulationConfig) -> Result<SimulationConfig> {
    println!("Droplet count [{}]: ", config.droplet_count);
    // For demo purposes, use defaults
    
    println!("Grid size (mm) [{}]: ", config.grid_size_mm);
    // Use defaults
    
    println!("Simulation speed multiplier [{}]: ", config.simulation_speed);
    // Use defaults
    
    println!("✅ Using default configuration");
    Ok(config)
}

async fn display_simulation_status(simulation: &simulation::MitochondriaSimulation) {
    println!("📊 Mitochondria v2 Status:");
    println!("  💧 Active droplets: {}", simulation.network.droplets.len());
    println!("  🧬 Total DNA mass: {:.2} pg", simulation.network.consensus_state.total_network_dna_mass);
    println!("  👑 Consensus leader: {}", simulation.network.consensus_state.heaviest_swarm_leader);
    println!("  🎯 Consensus confidence: {:.1}%", simulation.network.consensus_state.consensus_confidence * 100.0);
    println!("  🧅 Tor circuits: {}", simulation.network.tor_command_center.active_circuits);
    println!("  ⚡ Command queue: {} pending", simulation.network.tor_command_center.command_queue.len());
}

async fn spawn_new_droplet(simulation: &mut simulation::MitochondriaSimulation) -> Result<()> {
    let new_index = simulation.network.droplets.len();
    let new_droplet = create_genesis_droplet(new_index, &simulation.config).await?;
    let droplet_id = new_droplet.droplet_id.clone();
    
    simulation.network.droplets.insert(droplet_id.clone(), new_droplet);
    
    println!("🆕 Spawned new water-robot: {}", droplet_id);
    Ok(())
}

async fn execute_tor_command(
    simulation: &mut simulation::MitochondriaSimulation, 
    tor_cmd: &str
) -> Result<()> {
    println!("🧅 Executing Tor command: {}", tor_cmd);
    
    // Parse Tor command
    let parts: Vec<&str> = tor_cmd.split_whitespace().collect();
    
    if parts.is_empty() {
        println!("❓ Empty Tor command");
        return Ok(());
    }
    
    let command_type = match parts[0] {
        "move" => {
            if parts.len() >= 3 {
                let direction = match parts[1] {
                    "north" => Direction::North,
                    "south" => Direction::South,
                    "east" => Direction::East,
                    "west" => Direction::West,
                    _ => Direction::North,
                };
                let distance = parts[2].parse::<f64>().unwrap_or(100.0);
                CommandType::Move { direction, distance_um: distance }
            } else {
                CommandType::Move { direction: Direction::North, distance_um: 100.0 }
            }
        },
        
        "fission" => CommandType::InitiateFission,
        
        "synthesize" => CommandType::SynthesizeBlock { 
            block_data: parts[1..].join(" ").as_bytes().to_vec() 
        },
        
        _ => {
            println!("❓ Unknown Tor command: {}", parts[0]);
            return Ok(());
        }
    };
    
    // Create and queue Tor command for random droplet
    let target_droplet = simulation.network.droplets.keys().next()
        .cloned()
        .unwrap_or_default();
    
    let command = TorCommand {
        command_id: uuid::Uuid::new_v4().to_string(),
        target_droplet,
        command_type,
        payload: Vec::new(),
        issued_at: chrono::Utc::now(),
        executed: false,
    };
    
    simulation.network.tor_command_center.command_queue.push(command);
    
    println!("✅ Tor command queued for execution");
    Ok(())
}

async fn run_demo_sequence(simulation: &mut simulation::MitochondriaSimulation) -> Result<()> {
    println!("🎪 Running Mitochondria v2 demo sequence...");
    
    // Demo 1: Mass droplet movement
    println!("📦 Demo 1: Coordinated swarm movement");
    for droplet_id in simulation.network.droplets.keys().take(10).cloned().collect::<Vec<_>>() {
        let move_command = TorCommand {
            command_id: uuid::Uuid::new_v4().to_string(),
            target_droplet: droplet_id,
            command_type: CommandType::Move { 
                direction: Direction::Northeast, 
                distance_um: 200.0 
            },
            payload: Vec::new(),
            issued_at: chrono::Utc::now(),
            executed: false,
        };
        simulation.network.tor_command_center.command_queue.push(move_command);
    }
    
    // Demo 2: DNA synthesis wave
    println!("🧬 Demo 2: Coordinated DNA synthesis");
    for droplet_id in simulation.network.droplets.keys().take(5).cloned().collect::<Vec<_>>() {
        let synth_command = TorCommand {
            command_id: uuid::Uuid::new_v4().to_string(),
            target_droplet: droplet_id,
            command_type: CommandType::SynthesizeBlock { 
                block_data: b"DEMO_BLOCK_DATA_MITOCHONDRIA_V2".to_vec() 
            },
            payload: Vec::new(),
            issued_at: chrono::Utc::now(),
            executed: false,
        };
        simulation.network.tor_command_center.command_queue.push(synth_command);
    }
    
    // Demo 3: Binary fission
    println!("🔄 Demo 3: Triggered binary fission");
    if let Some(largest_droplet) = find_largest_droplet(&simulation.network.droplets) {
        let fission_command = TorCommand {
            command_id: uuid::Uuid::new_v4().to_string(),
            target_droplet: largest_droplet,
            command_type: CommandType::InitiateFission,
            payload: Vec::new(),
            issued_at: chrono::Utc::now(),
            executed: false,
        };
        simulation.network.tor_command_center.command_queue.push(fission_command);
    }
    
    println!("✅ Demo sequence queued - watch the magic happen!");
    Ok(())
}

fn find_largest_droplet(droplets: &HashMap<String, DropletNode>) -> Option<String> {
    droplets.iter()
        .max_by(|(_, a), (_, b)| a.size_nanoliters.partial_cmp(&b.size_nanoliters).unwrap())
        .map(|(id, _)| id.clone())
}

async fn display_detailed_stats(simulation: &simulation::MitochondriaSimulation) {
    println!("📈 Detailed Mitochondria v2 Statistics:");
    println!("  🧬 DNA Statistics:");
    
    let total_synthesis_events: usize = simulation.network.droplets.values()
        .map(|d| d.dna_data.synthesis_history.len())
        .sum();
    
    let average_chain_length: f64 = simulation.network.droplets.values()
        .map(|d| d.dna_data.chain_length as f64)
        .sum::<f64>() / simulation.network.droplets.len() as f64;
    
    println!("    Total synthesis events: {}", total_synthesis_events);
    println!("    Average chain length: {:.1} blocks", average_chain_length);
    println!("    Largest droplet mass: {:.2} pg", 
             simulation.network.droplets.values()
                 .map(|d| d.dna_data.total_mass_picograms)
                 .fold(0.0, f64::max));
    
    println!("  💧 Physics Statistics:");
    let total_energy: f64 = simulation.network.droplets.values()
        .map(|d| d.energy_level)
        .sum();
    
    let average_size: f64 = simulation.network.droplets.values()
        .map(|d| d.size_nanoliters)
        .sum::<f64>() / simulation.network.droplets.len() as f64;
    
    println!("    Total network energy: {:.2}", total_energy);
    println!("    Average droplet size: {:.1} nL", average_size);
    println!("    Grid utilization: {:.1}%", 
             (simulation.network.droplets.len() as f64 / 1000.0) * 100.0);
    
    println!("  🧅 Tor Statistics:");
    println!("    Active circuits: {}", simulation.network.tor_command_center.active_circuits);
    println!("    Commands queued: {}", simulation.network.tor_command_center.command_queue.len());
    println!("    Connected droplets: {}", simulation.network.tor_command_center.connected_droplets.len());
}

fn display_help() {
    println!("🧬💧 Mitochondria v2 Commands:");
    println!("  status     - Show simulation status");
    println!("  spawn      - Create new water-robot droplet");
    println!("  demo       - Run automated demo sequence");
    println!("  stats      - Show detailed statistics");
    println!("  tor <cmd>  - Send Tor command to droplets:");
    println!("    tor move north 200   - Move droplet north 200µm");
    println!("    tor fission          - Trigger binary fission");
    println!("    tor synthesize DATA  - Synthesize DNA block");
    println!("  quit       - Exit simulation");
    println!("🌟 The future of biological computing awaits!");
}

impl Clone for simulation::MitochondriaSimulation {
    fn clone(&self) -> Self {
        // Note: This is a simplified clone for demo purposes
        // In production, would need proper deep cloning
        Self {
            network: self.network.clone(),
            config: self.config.clone(),
            physics_engine: PhysicsEngine {
                surface_tension: 0.072,
                viscosity: 0.001, 
                electro_wetting_force: 1e-9,
                brownian_motion_coefficient: 1e-12,
            },
            dna_synthesizer: DNASynthesizer {
                synthesis_rate_pg_per_ms: 0.0001,
                error_rate: 0.001,
                energy_cost_per_base: 0.01,
                polymerase_efficiency: 0.95,
            },
            tor_interface: TorInterface {
                command_latency_ms: self.config.tor_command_latency_ms,
                packet_loss_rate: 0.01,
                encryption_overhead: 0.1,
            },
            visualization_state: VisualizationState {
                camera_position: nalgebra::Point2::new(5.0, 5.0),
                zoom_level: 1.0,
                tracked_droplets: Vec::new(),
                swarm_colors: std::collections::HashMap::new(),
            },
        }
    }
}

// Supporting structures for simulation engine
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
    camera_position: nalgebra::Point2<f64>,
    zoom_level: f64,
    tracked_droplets: Vec<String>,
    swarm_colors: std::collections::HashMap<String, (f32, f32, f32)>,
}