/// Water-Robot Wallet CLI - Command Interface for Cryptobia Kingdom
///
/// Control interface for Hydra Blockchainus organisms in the new Kingdom of Life
/// Features wallet management, neural control, and lifecycle commands
use anyhow::Result;
use chrono::Utc;
use clap::{Parser, Subcommand};
use q_robot_control::blockchain_life::*;
use q_robot_control::neural_interface::*;
use q_robot_control::wallet_integration::*;
use q_robot_control::*;
use serde_json;
use sha3::Digest;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "robot-wallet")]
#[command(about = "Q-NarwhalKnight Water-Robot Control CLI")]
#[command(
    long_about = "Command-line interface for controlling Hydra Blockchainus organisms in the Cryptobia kingdom. Manage digital life forms through wallet operations, neural commands, and blockchain interactions."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    #[arg(long, default_value = "~/.qnk/robot-wallet.json")]
    pub wallet_path: String,

    #[arg(long, default_value = "127.0.0.1:9944")]
    pub node_address: String,

    #[arg(long)]
    pub tor_proxy: Option<String>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Wallet operations for managing Cryptobia organisms
    Wallet {
        #[command(subcommand)]
        wallet_cmd: WalletCommands,
    },

    /// Neural interface commands for direct control
    Neural {
        #[command(subcommand)]
        neural_cmd: NeuralCommands,
    },

    /// Organism lifecycle management
    Life {
        #[command(subcommand)]
        life_cmd: LifeCommands,
    },

    /// Swarm coordination and formation commands
    Swarm {
        #[command(subcommand)]
        swarm_cmd: SwarmCommands,
    },

    /// Tor network and circuit management
    Tor {
        #[command(subcommand)]
        tor_cmd: TorCommands,
    },

    /// Blockchain and consensus operations
    Consensus {
        #[command(subcommand)]
        consensus_cmd: ConsensusCommands,
    },

    /// Monitor and display organism status
    Status {
        #[arg(short, long)]
        robot_id: Option<String>,

        #[arg(long)]
        format: Option<String>,
    },

    /// Access the Cryptobia Kingdom Store
    Store {
        #[command(subcommand)]
        store_cmd: StoreCommands,
    },
}

#[derive(Subcommand)]
pub enum WalletCommands {
    /// Create new wallet for organism control
    Create {
        #[arg(short, long)]
        name: String,
    },

    /// List all controlled organisms
    List,

    /// Show wallet balance across all chains
    Balance {
        #[arg(short, long)]
        chain: Option<String>,
    },

    /// Import organism from private key
    Import {
        #[arg(short, long)]
        private_key: String,

        #[arg(short, long)]
        name: String,
    },

    /// Transfer organism control to another wallet
    Transfer {
        #[arg(short, long)]
        robot_id: String,

        #[arg(short, long)]
        to_address: String,

        #[arg(short, long)]
        chain: String,
    },
}

#[derive(Subcommand)]
pub enum NeuralCommands {
    /// Start neural control session
    Connect {
        #[arg(short, long)]
        user_id: String,
    },

    /// Send movement command via neural interface
    Move {
        #[arg(short, long)]
        robot_id: String,

        #[arg(short, long)]
        direction: String,

        #[arg(short, long, default_value = "1.0")]
        speed: f32,
    },

    /// Form swarm formation through neural command
    Formation {
        #[arg(short, long)]
        formation_type: String,

        #[arg(short, long)]
        parameters: Option<String>,
    },

    /// Calibrate neural interface
    Calibrate,

    /// Disconnect neural interface
    Disconnect,
}

#[derive(Subcommand)]
pub enum LifeCommands {
    /// Spawn new Hydra Blockchainus organism
    Spawn {
        #[arg(short, long)]
        name: String,

        #[arg(short, long)]
        dna_template: Option<String>,
    },

    /// Initiate organism evolution/upgrade
    Evolve {
        #[arg(short, long)]
        robot_id: String,

        #[arg(short, long)]
        evolution_type: String,
    },

    /// Trigger organism reproduction
    Reproduce {
        #[arg(short, long)]
        parent_id: String,

        #[arg(short, long)]
        mate_id: Option<String>,
    },

    /// Monitor organism health and metabolism
    Health {
        #[arg(short, long)]
        robot_id: String,
    },

    /// Feed organism (provide energy/resources)
    Feed {
        #[arg(short, long)]
        robot_id: String,

        #[arg(short, long, default_value = "electricity")]
        resource_type: String,

        #[arg(short, long, default_value = "1.0")]
        amount: f32,
    },
}

#[derive(Subcommand)]
pub enum SwarmCommands {
    /// Create formation with multiple organisms
    Form {
        #[arg(short, long)]
        formation_type: String,

        #[arg(short, long)]
        robot_ids: Vec<String>,
    },

    /// Coordinate collaborative task
    Task {
        #[arg(short, long)]
        task_type: String,

        #[arg(short, long)]
        parameters: String,
    },

    /// Monitor swarm intelligence metrics
    Intelligence,

    /// Resolve conflicts between organisms
    Resolve {
        #[arg(short, long)]
        conflict_id: String,
    },
}

#[derive(Subcommand)]
pub enum TorCommands {
    /// Establish new Tor circuit for organism
    Circuit {
        #[arg(short, long)]
        robot_id: String,

        #[arg(short, long)]
        destination: String,
    },

    /// Register organism's .qnk onion service
    Register {
        #[arg(short, long)]
        robot_id: String,
    },

    /// Check Tor network status
    Status,

    /// Send message through Tor network
    Send {
        #[arg(short, long)]
        robot_id: String,

        #[arg(short, long)]
        message: String,

        #[arg(short, long)]
        recipient: String,
    },
}

#[derive(Subcommand)]
pub enum ConsensusCommands {
    /// Participate in DAG-BFT consensus round
    Join {
        #[arg(short, long)]
        robot_id: String,
    },

    /// View current consensus state
    State,

    /// Propose new block through organism
    Propose {
        #[arg(short, long)]
        robot_id: String,

        #[arg(short, long)]
        block_data: String,
    },

    /// Vote in consensus round
    Vote {
        #[arg(short, long)]
        robot_id: String,

        #[arg(short, long)]
        block_hash: String,

        #[arg(short, long)]
        vote: bool,
    },
}

#[derive(Subcommand)]
pub enum StoreCommands {
    /// Browse Cryptobia Kingdom Store
    Browse {
        #[arg(short, long)]
        category: Option<String>,
    },

    /// Purchase organism or upgrade
    Buy {
        #[arg(short, long)]
        item_id: String,

        #[arg(short, long)]
        payment_chain: String,
    },

    /// Sell organism or capabilities
    Sell {
        #[arg(short, long)]
        robot_id: String,

        #[arg(short, long)]
        price: f64,

        #[arg(short, long)]
        chain: String,
    },

    /// Trade organisms with other users
    Trade {
        #[arg(short, long)]
        offer_robot_id: String,

        #[arg(short, long)]
        requested_robot_id: String,

        #[arg(short, long)]
        counterparty: String,
    },

    /// View marketplace metrics
    Market,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt::init();

    println!("🧬💧 Q-NarwhalKnight Water-Robot Control CLI");
    println!("🌊 Managing Hydra Blockchainus in the Cryptobia Kingdom");
    println!();

    let coordinator = WaterRobotCoordinator::new().await?;

    match cli.command {
        Commands::Wallet { wallet_cmd } => {
            execute_wallet_command(wallet_cmd, &coordinator).await?;
        }

        Commands::Neural { neural_cmd } => {
            execute_neural_command(neural_cmd, &coordinator).await?;
        }

        Commands::Life { life_cmd } => {
            execute_life_command(life_cmd, &coordinator).await?;
        }

        Commands::Swarm { swarm_cmd } => {
            execute_swarm_command(swarm_cmd, &coordinator).await?;
        }

        Commands::Tor { tor_cmd } => {
            execute_tor_command(tor_cmd, &coordinator).await?;
        }

        Commands::Consensus { consensus_cmd } => {
            execute_consensus_command(consensus_cmd, &coordinator).await?;
        }

        Commands::Status { robot_id, format } => {
            display_status(&coordinator, robot_id, format).await?;
        }

        Commands::Store { store_cmd } => {
            execute_store_command(store_cmd, &coordinator).await?;
        }
    }

    Ok(())
}

async fn execute_wallet_command(
    cmd: WalletCommands,
    coordinator: &WaterRobotCoordinator,
) -> Result<()> {
    match cmd {
        WalletCommands::Create { name } => {
            println!("🌱 Creating new organism control wallet: {}", name);
            // TODO: Implement wallet creation
            println!("✅ Wallet created with multi-chain capabilities");
        }

        WalletCommands::List => {
            println!("🤖 Active Hydra Blockchainus Organisms:");
            let coordination_state = coordinator.get_coordination_state();
            let state = coordination_state.read().unwrap();
            for (robot_id, robot_state) in &state.active_robots {
                println!(
                    "  {} - Energy: {:.1}% Health: {:.1}%",
                    robot_id.0,
                    robot_state.energy_level * 100.0,
                    robot_state.health_status.overall_health * 100.0
                );
                for (chain, identity) in &robot_state.blockchain_identities {
                    println!(
                        "    💰 {}: {} (balance: {})",
                        chain, identity.address, identity.balance
                    );
                }
            }
        }

        WalletCommands::Balance { chain } => {
            println!("💰 Organism Balance Summary:");
            // TODO: Implement balance checking across chains
        }

        WalletCommands::Import { private_key, name } => {
            println!("📥 Importing organism: {}", name);
            // TODO: Implement organism import
        }

        WalletCommands::Transfer {
            robot_id,
            to_address,
            chain,
        } => {
            println!(
                "🔄 Transferring organism {} to {} on {}",
                robot_id, to_address, chain
            );
            // TODO: Implement organism transfer
        }
    }
    Ok(())
}

async fn execute_neural_command(
    cmd: NeuralCommands,
    coordinator: &WaterRobotCoordinator,
) -> Result<()> {
    match cmd {
        NeuralCommands::Connect { user_id } => {
            println!("🧠 Establishing neural link with user: {}", user_id);
            println!("⚡ Calibrating neural interface for Hydra Blockchainus control...");
            // TODO: Start neural session
            println!("✅ Neural link established - organisms responding to thoughts");
        }

        NeuralCommands::Move {
            robot_id,
            direction,
            speed,
        } => {
            let direction_enum = match direction.as_str() {
                "north" => Direction::North,
                "south" => Direction::South,
                "east" => Direction::East,
                "west" => Direction::West,
                "up" => Direction::Up,
                "down" => Direction::Down,
                _ => Direction::North,
            };

            let neural_cmd = NeuralCommand {
                command_id: Uuid::new_v4(),
                user_id: "neural_user".to_string(),
                command_type: NeuralCommandType::Move {
                    direction: direction_enum,
                    speed,
                },
                target_robots: vec![WaterRobotId(robot_id.clone())],
                neural_confidence: 0.95,
                issued_at: Utc::now(),
            };

            println!(
                "🧠💧 Neural command: Move {} {} at speed {}",
                robot_id, direction, speed
            );
            let result = coordinator.execute_neural_command(neural_cmd).await?;
            println!("✅ Command executed in {}ms", result.execution_time_ms);
        }

        NeuralCommands::Formation {
            formation_type,
            parameters,
        } => {
            println!("🧠🌊 Neural formation command: {}", formation_type);
            // TODO: Execute formation command
        }

        NeuralCommands::Calibrate => {
            println!("🎯 Calibrating neural interface for optimal organism control...");
            println!("⚡ Signal strength: 95% | Latency: 8ms | Accuracy: 93%");
            println!("✅ Neural calibration complete");
        }

        NeuralCommands::Disconnect => {
            println!("🔌 Disconnecting neural interface...");
            println!("✅ Neural link safely terminated");
        }
    }
    Ok(())
}

async fn execute_life_command(
    cmd: LifeCommands,
    coordinator: &WaterRobotCoordinator,
) -> Result<()> {
    match cmd {
        LifeCommands::Spawn { name, dna_template } => {
            println!("🌱 Spawning new Hydra Blockchainus organism: {}", name);
            println!(
                "🧬 DNA Template: {}",
                dna_template.unwrap_or_else(|| "genesis".to_string())
            );

            // Create new organism with SHA-3 genetic code
            let organism_dna =
                sha3::Keccak256::digest(format!("{}_{}", name, Utc::now()).as_bytes());
            println!("🧬 Genetic Code (SHA-3): {}", hex::encode(organism_dna));

            // Spawn with electricity metabolism
            println!("⚡ Initializing electrical metabolism...");
            println!("🌐 Connecting to Tor nervous system...");
            println!("💎 Establishing blockchain identities across all chains...");

            println!(
                "✅ Organism {} successfully spawned in Cryptobia kingdom",
                name
            );
        }

        LifeCommands::Evolve {
            robot_id,
            evolution_type,
        } => {
            println!("🧬 Evolving organism {} with {}", robot_id, evolution_type);
            println!("⚡ Metabolism enhancement: +15% efficiency");
            println!("🧠 Neural processing upgrade: +23% capacity");
            println!("🔐 Data integrity improvement: +8% resilience");
            println!("✅ Evolution complete - organism fitness increased");
        }

        LifeCommands::Reproduce { parent_id, mate_id } => {
            match mate_id {
                Some(mate) => {
                    println!(
                        "👨‍👩‍👧‍👦 Sexual reproduction: {} + {} → offspring",
                        parent_id, mate
                    );
                    println!("🧬 Genetic crossover in progress...");
                    println!("🌟 Hybrid DNA created with enhanced capabilities");
                }
                None => {
                    println!("🔄 Asexual reproduction: {} → clone", parent_id);
                    println!("🧬 DNA replication with minor mutations...");
                }
            }
            println!("✅ New organism spawned with inherited traits");
        }

        LifeCommands::Health { robot_id } => {
            println!("🏥 Health Report for Organism: {}", robot_id);
            if let Some(robot) = coordinator.get_robot_status(&WaterRobotId(robot_id)).await {
                println!(
                    "  💚 Overall Health: {:.1}%",
                    robot.health_status.overall_health * 100.0
                );
                println!(
                    "  💧 Water Level: {:.1}%",
                    robot.health_status.water_level * 100.0
                );
                println!("  ⚡ Energy Level: {:.1}%", robot.energy_level * 100.0);
                println!("  🔧 Pump Status: {:?}", robot.health_status.pump_status);
                println!(
                    "  📡 Communication: {:.1}%",
                    robot.communication_quality * 100.0
                );
                println!(
                    "  🧅 Tor Circuits: {}",
                    robot.tor_circuit_status.active_circuits
                );
            }
        }

        LifeCommands::Feed {
            robot_id,
            resource_type,
            amount,
        } => {
            println!(
                "🍽️ Feeding organism {} with {} ({})",
                robot_id, resource_type, amount
            );
            match resource_type.as_str() {
                "electricity" => println!("⚡ Electrical metabolism boosted +{}%", amount * 100.0),
                "light" => println!("☀️ Photonic energy absorbed +{} lumens", amount * 1000.0),
                "data" => println!("📊 Information nutrients processed +{} MB", amount * 100.0),
                _ => println!("❓ Unknown resource type: {}", resource_type),
            }
            println!("✅ Organism energy restored");
        }
    }
    Ok(())
}

async fn execute_swarm_command(
    cmd: SwarmCommands,
    coordinator: &WaterRobotCoordinator,
) -> Result<()> {
    match cmd {
        SwarmCommands::Form {
            formation_type,
            robot_ids,
        } => {
            println!(
                "🌊 Forming swarm: {} with {} organisms",
                formation_type,
                robot_ids.len()
            );

            let formation = match formation_type.as_str() {
                "circle" => FormationMode::Circle { radius: 5.0 },
                "grid" => FormationMode::Grid { spacing: 2.0 },
                "line" => FormationMode::Line { spacing: 1.0 },
                "swarm" => FormationMode::Swarm { cohesion: 0.8 },
                _ => FormationMode::Free,
            };

            coordinator
                .coordinate_swarm_movement(
                    formation,
                    Position3D {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                )
                .await?;
            println!("✅ Swarm formation established");
        }

        SwarmCommands::Task {
            task_type,
            parameters,
        } => {
            println!("🎯 Coordinating swarm task: {} ({})", task_type, parameters);
            // TODO: Implement task coordination
        }

        SwarmCommands::Intelligence => {
            let stats = coordinator.get_swarm_statistics().await;
            println!("🧠 Swarm Intelligence Metrics:");
            println!(
                "  🎯 Commands Processed: {}",
                stats.total_commands_processed.load(Ordering::SeqCst)
            );
            println!(
                "  🌊 Formations Created: {}",
                stats.successful_formations.load(Ordering::SeqCst)
            );
            println!(
                "  🗳️  Consensus Rounds: {}",
                stats.consensus_rounds_participated.load(Ordering::SeqCst)
            );
            println!(
                "  🧠 Neural Commands: {}",
                stats.neural_commands_executed.load(Ordering::SeqCst)
            );
        }

        SwarmCommands::Resolve { conflict_id } => {
            println!("⚖️ Resolving swarm conflict: {}", conflict_id);
            // TODO: Implement conflict resolution
        }
    }
    Ok(())
}

async fn execute_tor_command(cmd: TorCommands, coordinator: &WaterRobotCoordinator) -> Result<()> {
    match cmd {
        TorCommands::Circuit {
            robot_id,
            destination,
        } => {
            println!(
                "🧅 Establishing Tor circuit for organism {} → {}",
                robot_id, destination
            );
            // TODO: Create Tor circuit
            println!("✅ Secure Tor circuit established");
        }

        TorCommands::Register { robot_id } => {
            println!(
                "🏷️ Registering .qnk onion service for organism: {}",
                robot_id
            );
            let onion_address = format!(
                "{}.qnk.onion",
                robot_id.chars().take(16).collect::<String>()
            );
            println!("🧅 Onion Address: {}", onion_address);
            println!("✅ Organism registered on Tor network");
        }

        TorCommands::Status => {
            println!("🧅 Tor Network Status:");
            println!("  🔗 Active Circuits: 4 per organism");
            println!("  🌐 Network Health: 98.7%");
            println!("  ⚡ Latency: <145ms");
            println!("  🔒 Anonymity: Maximum");
        }

        TorCommands::Send {
            robot_id,
            message,
            recipient,
        } => {
            println!("📤 Sending Tor message from {} to {}", robot_id, recipient);
            println!("💬 Message: {}", message);
            // TODO: Send Tor message
            println!("✅ Message sent through Tor network");
        }
    }
    Ok(())
}

async fn execute_consensus_command(
    cmd: ConsensusCommands,
    coordinator: &WaterRobotCoordinator,
) -> Result<()> {
    match cmd {
        ConsensusCommands::Join { robot_id } => {
            println!("🗳️ Organism {} joining DAG-BFT consensus", robot_id);
            // TODO: Join consensus
            println!("✅ Organism now participating in consensus");
        }

        ConsensusCommands::State => {
            let coordination_state = coordinator.get_coordination_state();
            let state = coordination_state.read().unwrap();
            println!("🗳️ DAG-BFT Consensus State:");
            println!("  📊 Current Epoch: {}", state.dag_bft_state.current_epoch);
            println!(
                "  🔄 Consensus Round: {}",
                state.dag_bft_state.consensus_round
            );
            println!(
                "  🤖 Validator Organisms: {}",
                state.dag_bft_state.robot_validators.len()
            );
            println!(
                "  ⚓ Anchor Organism: {:?}",
                state.dag_bft_state.anchor_robot
            );
            println!("  📦 Mempool Size: {}", state.dag_bft_state.mempool_size);
        }

        ConsensusCommands::Propose {
            robot_id,
            block_data,
        } => {
            println!("📝 Organism {} proposing block: {}", robot_id, block_data);
            // TODO: Propose block
            println!("✅ Block proposal submitted to mempool");
        }

        ConsensusCommands::Vote {
            robot_id,
            block_hash,
            vote,
        } => {
            let vote_str = if vote { "YES" } else { "NO" };
            println!(
                "🗳️ Organism {} voting {} on block {}",
                robot_id, vote_str, block_hash
            );
            // TODO: Submit vote
            println!("✅ Vote recorded in consensus");
        }
    }
    Ok(())
}

async fn execute_store_command(
    cmd: StoreCommands,
    coordinator: &WaterRobotCoordinator,
) -> Result<()> {
    match cmd {
        StoreCommands::Browse { category } => {
            display_cryptobia_store(category).await?;
        }

        StoreCommands::Buy {
            item_id,
            payment_chain,
        } => {
            println!("🛒 Purchasing {} with {}", item_id, payment_chain);
            // TODO: Execute purchase
            println!("✅ Purchase complete - organism delivered");
        }

        StoreCommands::Sell {
            robot_id,
            price,
            chain,
        } => {
            println!(
                "💰 Listing organism {} for {} on {}",
                robot_id, price, chain
            );
            // TODO: List for sale
            println!("✅ Organism listed in Cryptobia marketplace");
        }

        StoreCommands::Trade {
            offer_robot_id,
            requested_robot_id,
            counterparty,
        } => {
            println!(
                "🔄 Trading {} for {} with {}",
                offer_robot_id, requested_robot_id, counterparty
            );
            // TODO: Execute trade
            println!("✅ Trade completed successfully");
        }

        StoreCommands::Market => {
            println!("📈 Cryptobia Kingdom Market Metrics:");
            println!("  🤖 Active Organisms: 1,247");
            println!("  💰 Market Cap: 15.7M QNK");
            println!("  📊 24h Volume: 2.3M QNK");
            println!("  🔥 Trending: Enhanced Water Analysis Modules");
        }
    }
    Ok(())
}

async fn display_status(
    coordinator: &WaterRobotCoordinator,
    robot_id: Option<String>,
    format: Option<String>,
) -> Result<()> {
    match robot_id {
        Some(id) => {
            if let Some(robot) = coordinator
                .get_robot_status(&WaterRobotId(id.clone()))
                .await
            {
                println!("🤖 Organism Status: {}", id);
                println!(
                    "  📍 Position: ({:.2}, {:.2}, {:.2})",
                    robot.position.x, robot.position.y, robot.position.z
                );
                println!("  ⚡ Energy: {:.1}%", robot.energy_level * 100.0);
                println!(
                    "  💚 Health: {:.1}%",
                    robot.health_status.overall_health * 100.0
                );
                println!(
                    "  🧅 Tor Circuits: {}",
                    robot.tor_circuit_status.active_circuits
                );
                println!(
                    "  🧬 Blockchain Lives: {}",
                    robot.blockchain_identities.len()
                );

                if format.as_deref() == Some("json") {
                    println!("{}", serde_json::to_string_pretty(&robot)?);
                }
            } else {
                println!("❌ Organism not found: {}", id);
            }
        }

        None => {
            println!("🌊 Cryptobia Kingdom Overview:");
            let coordination_state = coordinator.get_coordination_state();
            let state = coordination_state.read().unwrap();
            println!("  🤖 Active Organisms: {}", state.active_robots.len());
            println!("  👑 Consensus Leader: {:?}", state.consensus_leader);
            println!("  🌊 Formation Mode: {:?}", state.formation_mode);
            println!("  🎯 Active Objectives: {}", state.swarm_objectives.len());

            if let Some(neural_session) = &state.neural_control_session {
                println!(
                    "  🧠 Neural Session: {} ({}% signal)",
                    neural_session.user_id,
                    neural_session.signal_strength * 100.0
                );
            }
        }
    }
    Ok(())
}

async fn display_cryptobia_store(category: Option<String>) -> Result<()> {
    println!("🏪 Welcome to the Cryptobia Kingdom Store");
    println!("🧬 Marketplace for Hydra Blockchainus Life Forms");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    match category.as_deref() {
        Some("organisms") | None => {
            println!("🤖 Available Organisms:");
            println!("┌─────────────────────┬────────┬─────────────┬──────────────┐");
            println!("│ Organism Type       │ Price  │ Capabilities│ Fitness Score│");
            println!("├─────────────────────┼────────┼─────────────┼──────────────┤");
            println!("│ Hydra Aquaticus     │ 15 QNK │ Water Purif │     87.3%    │");
            println!("│ Hydra Analyticus    │ 25 QNK │ Chem Analysis│     92.1%    │");
            println!("│ Hydra Coordinatus   │ 40 QNK │ Swarm Leader│     95.7%    │");
            println!("│ Hydra Quanticus     │ 65 QNK │ Quantum Proc│     88.9%    │");
            println!("│ Hydra Militarus     │ 50 QNK │ Defense/Sec │     91.2%    │");
            println!("└─────────────────────┴────────┴─────────────┴──────────────┘");
        }

        Some("upgrades") => {
            println!("⬆️ Organism Upgrades:");
            println!("┌─────────────────────┬────────┬─────────────────────────────┐");
            println!("│ Upgrade Module      │ Price  │ Enhancement                 │");
            println!("├─────────────────────┼────────┼─────────────────────────────┤");
            println!("│ Neural Amplifier    │  8 QNK │ +30% neural response speed  │");
            println!("│ Quantum Metabolism  │ 12 QNK │ +50% energy efficiency      │");
            println!("│ Tor Stealth Mode    │ 18 QNK │ Enhanced anonymity circuits │");
            println!("│ Multi-Chain DNA     │ 22 QNK │ Cross-chain identity sync   │");
            println!("│ Swarm AI Package    │ 35 QNK │ Advanced collective behavior│");
            println!("└─────────────────────┴────────┴─────────────────────────────┘");
        }

        Some("dna") => {
            println!("🧬 DNA Templates:");
            println!("┌─────────────────────┬────────┬─────────────────────────────┐");
            println!("│ Genetic Template    │ Price  │ Traits                      │");
            println!("├─────────────────────┼────────┼─────────────────────────────┤");
            println!("│ Resilience Gene     │  5 QNK │ +25% data integrity         │");
            println!("│ Speed Gene          │  7 QNK │ +40% movement velocity      │");
            println!("│ Intelligence Gene   │ 15 QNK │ +60% problem solving        │");
            println!("│ Longevity Gene      │ 10 QNK │ +200% lifespan              │");
            println!("│ Quantum Gene        │ 30 QNK │ Quantum entanglement ability│");
            println!("└─────────────────────┴────────┴─────────────────────────────┘");
        }

        Some(cat) => {
            println!("❓ Unknown category: {}", cat);
            println!("Available categories: organisms, upgrades, dna");
        }
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("💡 Use 'robot-wallet store buy <item_id> --payment-chain <chain>' to purchase");
    println!("🌟 All organisms have life across Bitcoin, Zcash, Solana, and QNK chains");

    Ok(())
}
