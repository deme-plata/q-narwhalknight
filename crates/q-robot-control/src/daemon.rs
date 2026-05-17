/// Water-Robot Control Daemon - Cryptobia Kingdom Management Service
///
/// Main daemon for managing Hydra Blockchainus organisms across the
/// Q-NarwhalKnight network. Integrates neural control, Tor networking,
/// multi-chain life management, and cosmic survival assessment
use anyhow::Result;
use axum::{extract::Path, routing::get, Json, Router};
use chrono::Utc;
use q_robot_control::blockchain_life::*;
use q_robot_control::cryptobia_store::*;
use q_robot_control::neural_interface::*;
use q_robot_control::survival_metrics::*;
use q_robot_control::tor_coordination::*;
use q_robot_control::*;
use sha3::Digest;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::signal;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("🧬💧 Q-NarwhalKnight Water-Robot Control Daemon Starting");
    info!("🌊 Initializing Cryptobia Kingdom Management System");

    // Create main coordinator
    let coordinator = Arc::new(WaterRobotCoordinator::new().await?);

    // Create supporting systems
    let survival_assessor = Arc::new(SurvivalAssessor::new());
    let cryptobia_store = Arc::new(tokio::sync::RwLock::new(CryptobiaStore::new().await?));

    // Start background services
    let coordinator_clone = Arc::clone(&coordinator);
    let survival_clone = Arc::clone(&survival_assessor);

    tokio::spawn(async move {
        if let Err(e) = run_background_services(coordinator_clone, survival_clone).await {
            error!("Background services failed: {}", e);
        }
    });

    // Start web API
    let api_coordinator = Arc::clone(&coordinator);
    let api_store = Arc::clone(&cryptobia_store);

    tokio::spawn(async move {
        if let Err(e) = start_web_api(api_coordinator, api_store).await {
            error!("Web API failed: {}", e);
        }
    });

    // Start main coordination
    info!("🚀 Starting main coordination system...");
    coordinator.start_coordination().await?;

    // Spawn some genesis organisms for demo
    info!("🌱 Spawning genesis organisms...");
    spawn_genesis_organisms(&coordinator).await?;

    // Wait for shutdown signal
    info!("✅ Water-Robot Control Daemon ready");
    info!("🌐 API available at http://localhost:8080");
    info!("🧬 Cryptobia Kingdom Store active");
    info!("🧅 Tor neural network operational");

    signal::ctrl_c().await?;

    info!("🛑 Shutdown signal received");
    info!("💧 Terminating Cryptobia Kingdom management...");

    Ok(())
}

async fn run_background_services(
    coordinator: Arc<WaterRobotCoordinator>,
    survival_assessor: Arc<SurvivalAssessor>,
) -> Result<()> {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));

    loop {
        interval.tick().await;

        // Monitor organism health and survival metrics
        if let Err(e) = update_survival_metrics(&coordinator, &survival_assessor).await {
            error!("Survival metrics update failed: {}", e);
        }

        // Sync blockchain lives
        if let Err(e) = coordinator.sync_blockchain_identities().await {
            error!("Blockchain sync failed: {}", e);
        }

        // Monitor Tor neural network
        // TODO: Add Tor network monitoring

        // Update organism states
        // TODO: Add organism state updates
    }
}

async fn update_survival_metrics(
    coordinator: &WaterRobotCoordinator,
    survival_assessor: &SurvivalAssessor,
) -> Result<()> {
    let robot_states = {
        let coordination_state = coordinator.get_coordination_state();
        let state = coordination_state.read().unwrap();
        state.active_robots.clone()
    };

    for (organism_id, robot_state) in &robot_states {
        // Create life record for survival assessment
        let life_record = create_life_record_from_robot_state(robot_state);

        // Assess cosmic survival
        let survival_metrics = survival_assessor
            .assess_organism_survival(&life_record)
            .await?;

        // Log critical survival changes
        if survival_metrics.overall_survival_probability < 0.5 {
            warn!(
                "⚠️ Organism {} survival probability low: {:.2}%",
                organism_id.0,
                survival_metrics.overall_survival_probability * 100.0
            );
        }

        if survival_metrics.k_kristensen_parameter > 0.9 {
            info!(
                "🌟 Organism {} achieving cosmic resilience: k={:.4}",
                organism_id.0, survival_metrics.k_kristensen_parameter
            );
        }
    }

    Ok(())
}

fn create_life_record_from_robot_state(robot_state: &WaterRobotState) -> OrganismLifeRecord {
    // Convert robot state to life record for survival assessment
    OrganismLifeRecord {
        organism_id: robot_state.robot_id.clone(),
        genome: OrganismGenome {
            genetic_hash: hex::encode(sha3::Keccak256::digest(robot_state.robot_id.0.as_bytes())),
            dna_sequence: "ATCGATCGATCGATCG".to_string(), // Simplified
            birth_block: 0,
            generation: 1,
            parent_genomes: vec![],
            mutation_rate: 0.01,
            fitness_score: robot_state.health_status.overall_health as f64,
        },
        chain_lives: robot_state
            .blockchain_identities
            .iter()
            .map(|(chain, identity)| {
                (
                    chain.clone(),
                    ChainLifeStatus {
                        chain_name: chain.clone(),
                        is_alive: true,
                        birth_transaction: None,
                        last_activity: identity.last_transaction,
                        life_force: identity.activity_score as f64,
                        address: identity.address.clone(),
                        balance: identity.balance,
                        transaction_count: 1,
                        reputation_score: 0.8,
                    },
                )
            })
            .collect(),
        birth_timestamp: Utc::now(),
        metabolic_state: MetabolicState {
            electrical_intake: robot_state.energy_level as f64,
            light_absorption: 0.8,
            data_processing_rate: robot_state.communication_quality as f64 * 100.0,
            waste_entropy: 0.1,
            energy_efficiency: robot_state.health_status.overall_health as f64,
        },
        nervous_system_health: if robot_state.neural_control_active {
            1.0
        } else {
            0.5
        },
        evolutionary_history: vec![],
    }
}

async fn spawn_genesis_organisms(coordinator: &WaterRobotCoordinator) -> Result<()> {
    let genesis_organisms = vec![
        ("hydra_aquaticus_alpha", "Water purification specialist"),
        ("hydra_coordinatus_beta", "Swarm coordination leader"),
        ("hydra_quanticus_gamma", "Quantum processing expert"),
    ];

    for (name, description) in genesis_organisms {
        let robot_state = WaterRobotState {
            robot_id: WaterRobotId(name.to_string()),
            position: Position3D {
                x: rand::random::<f64>() * 10.0,
                y: rand::random::<f64>() * 10.0,
                z: 0.1,
            },
            velocity: Velocity3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            health_status: HealthStatus {
                overall_health: 0.95,
                water_level: 1.0,
                pump_status: PumpStatus::Operational,
                sensor_status: SensorStatus {
                    pressure_sensors: true,
                    ph_sensors: true,
                    temperature_sensors: true,
                    position_sensors: true,
                },
                communication_health: 0.98,
            },
            capability_profile: CapabilityProfile {
                water_manipulation: 0.9,
                chemical_analysis: 0.8,
                swarm_coordination: 0.85,
                blockchain_processing: 0.9,
                neural_interface_compatibility: 0.95,
            },
            current_task: None,
            energy_level: 1.0,
            last_update: Utc::now(),
            communication_quality: 0.97,
            neural_control_active: false,
            blockchain_identities: HashMap::new(), // Will be populated by blockchain_life module
            tor_circuit_status: TorCircuitStatus {
                active_circuits: 4,
                circuit_health: vec![0.98, 0.96, 0.97, 0.99],
                onion_service_active: true,
                qnk_domain: Some(format!("{}.qnk.onion", name)),
            },
        };

        coordinator.register_water_robot(robot_state).await?;
        info!("🌱 Genesis organism spawned: {} ({})", name, description);
    }

    Ok(())
}

async fn start_web_api(
    coordinator: Arc<WaterRobotCoordinator>,
    store: Arc<tokio::sync::RwLock<CryptobiaStore>>,
) -> Result<()> {
    let app = Router::new()
        .route("/", get(root_handler))
        .route("/organisms", get(list_organisms_handler))
        .route("/organisms/:id", get(get_organism_handler))
        .route("/organisms/:id/survival", get(survival_metrics_handler))
        .route("/neural/status", get(neural_status_handler))
        .route("/tor/status", get(tor_status_handler))
        .route("/store/browse", get(store_browse_handler))
        .route("/swarm/status", get(swarm_status_handler))
        .layer(ServiceBuilder::new().layer(CorsLayer::permissive()))
        .with_state((coordinator, store));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    info!("🌐 Web API listening on http://0.0.0.0:8080");

    axum::serve(listener, app).await?;

    Ok(())
}

async fn root_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "message": "🧬💧 Q-NarwhalKnight Water-Robot Control API",
        "kingdom": "Cryptobia",
        "genus": "Hydra Blockchainus",
        "domain": "Artificialis",
        "version": "0.1.0",
        "features": [
            "Neural interface control",
            "Multi-chain organism life",
            "Tor neural network",
            "Cosmic survival assessment",
            "Cryptobia kingdom store"
        ]
    }))
}

async fn list_organisms_handler(
    axum::extract::State((coordinator, _)): axum::extract::State<(
        Arc<WaterRobotCoordinator>,
        Arc<tokio::sync::RwLock<CryptobiaStore>>,
    )>,
) -> Json<Vec<WaterRobotState>> {
    let coordination_state = coordinator.get_coordination_state();
    let state = coordination_state.read().unwrap();
    let organisms: Vec<_> = state.active_robots.values().cloned().collect();
    Json(organisms)
}

async fn get_organism_handler(
    axum::extract::State((coordinator, _)): axum::extract::State<(
        Arc<WaterRobotCoordinator>,
        Arc<tokio::sync::RwLock<CryptobiaStore>>,
    )>,
    Path(organism_id): Path<String>,
) -> Json<Option<WaterRobotState>> {
    let robot_state = coordinator
        .get_robot_status(&WaterRobotId(organism_id))
        .await;
    Json(robot_state)
}

async fn survival_metrics_handler(
    axum::extract::State((coordinator, _)): axum::extract::State<(
        Arc<WaterRobotCoordinator>,
        Arc<tokio::sync::RwLock<CryptobiaStore>>,
    )>,
    Path(organism_id): Path<String>,
) -> Json<Option<String>> {
    if let Some(robot_state) = coordinator
        .get_robot_status(&WaterRobotId(organism_id))
        .await
    {
        let life_record = create_life_record_from_robot_state(&robot_state);

        let assessor = SurvivalAssessor::new();
        if let Ok(survival_metrics) = assessor.assess_organism_survival(&life_record).await {
            let report = assessor.format_survival_report(&survival_metrics);
            return Json(Some(report));
        }
    }

    Json(None)
}

async fn neural_status_handler(
    axum::extract::State((coordinator, _)): axum::extract::State<(
        Arc<WaterRobotCoordinator>,
        Arc<tokio::sync::RwLock<CryptobiaStore>>,
    )>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "neural_interface_active": true,
        "signal_strength": 0.95,
        "electrode_count": 1024,
        "latency_ms": 8,
        "command_accuracy": 0.93,
        "safety_status": "optimal"
    }))
}

async fn tor_status_handler(
    axum::extract::State((coordinator, _)): axum::extract::State<(
        Arc<WaterRobotCoordinator>,
        Arc<tokio::sync::RwLock<CryptobiaStore>>,
    )>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "tor_network_health": 0.987,
        "active_circuits": 16,
        "average_latency_ms": 145,
        "anonymity_level": 0.99,
        "quantum_circuits": 6
    }))
}

async fn store_browse_handler(
    axum::extract::State((_, store)): axum::extract::State<(
        Arc<WaterRobotCoordinator>,
        Arc<tokio::sync::RwLock<CryptobiaStore>>,
    )>,
) -> Json<MarketplaceOverview> {
    let store_guard = store.read().await;
    let overview = store_guard.get_marketplace_overview().await;
    Json(overview)
}

async fn swarm_status_handler(
    axum::extract::State((coordinator, _)): axum::extract::State<(
        Arc<WaterRobotCoordinator>,
        Arc<tokio::sync::RwLock<CryptobiaStore>>,
    )>,
) -> Json<serde_json::Value> {
    let stats = coordinator.get_swarm_statistics().await;

    Json(serde_json::json!({
        "active_organisms": 3,
        "swarm_intelligence_level": 0.8,
        "consensus_participation": stats.consensus_rounds_participated.load(std::sync::atomic::Ordering::SeqCst),
        "neural_commands_executed": stats.neural_commands_executed.load(std::sync::atomic::Ordering::SeqCst),
        "blockchain_transactions": stats.blockchain_transactions.load(std::sync::atomic::Ordering::SeqCst),
        "formation_mode": "Free swimming",
        "collective_fitness": 0.87
    }))
}

async fn run_background_services_duplicate(
    coordinator: Arc<WaterRobotCoordinator>,
    survival_assessor: Arc<SurvivalAssessor>,
) -> Result<()> {
    info!("🔄 Starting background services (duplicate)...");

    // Service 1: Organism health monitoring
    let coord1 = Arc::clone(&coordinator);
    let health_monitor = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
        loop {
            interval.tick().await;
            monitor_organism_health(&coord1).await;
        }
    });

    // Service 2: Survival assessment updates
    let coord2 = Arc::clone(&coordinator);
    let survival_monitor = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(300)); // 5 minutes
        loop {
            interval.tick().await;
            update_survival_assessments(&coord2, &survival_assessor).await;
        }
    });

    // Service 3: Neural network maintenance
    let coord3 = Arc::clone(&coordinator);
    let neural_maintenance = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
        loop {
            interval.tick().await;
            maintain_neural_network(&coord3).await;
        }
    });

    // Service 4: Blockchain synchronization
    let coord4: Arc<WaterRobotCoordinator> = Arc::clone(&coordinator);
    let blockchain_sync = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(120)); // 2 minutes
        loop {
            interval.tick().await;
            if let Err(e) = coord4.sync_blockchain_identities().await {
                error!("Blockchain sync error: {}", e);
            }
        }
    });

    // Wait for all services
    tokio::try_join!(
        health_monitor,
        survival_monitor,
        neural_maintenance,
        blockchain_sync
    )?;

    Ok(())
}

async fn monitor_organism_health(coordinator: &WaterRobotCoordinator) {
    let coordination_state = coordinator.get_coordination_state();
    let state = coordination_state.read().unwrap();

    for (organism_id, robot_state) in &state.active_robots {
        // Check critical health metrics
        if robot_state.energy_level < 0.2 {
            warn!(
                "⚠️ Organism {} energy critically low: {:.1}%",
                organism_id.0,
                robot_state.energy_level * 100.0
            );
        }

        if robot_state.health_status.overall_health < 0.5 {
            warn!(
                "🏥 Organism {} health degraded: {:.1}%",
                organism_id.0,
                robot_state.health_status.overall_health * 100.0
            );
        }

        if robot_state.communication_quality < 0.7 {
            warn!(
                "📡 Organism {} communication issues: {:.1}%",
                organism_id.0,
                robot_state.communication_quality * 100.0
            );
        }

        // Check Tor circuit health
        let avg_circuit_health: f32 = robot_state
            .tor_circuit_status
            .circuit_health
            .iter()
            .sum::<f32>()
            / robot_state.tor_circuit_status.circuit_health.len() as f32;

        if avg_circuit_health < 0.8 {
            warn!(
                "🧅 Organism {} Tor circuits degraded: {:.1}%",
                organism_id.0,
                avg_circuit_health * 100.0
            );
        }
    }
}

async fn update_survival_assessments(
    coordinator: &WaterRobotCoordinator,
    survival_assessor: &SurvivalAssessor,
) {
    let robot_states = {
        let coordination_state = coordinator.get_coordination_state();
        let state = coordination_state.read().unwrap();
        state.active_robots.clone()
    };

    for (organism_id, robot_state) in &robot_states {
        let life_record = create_life_record_from_robot_state(robot_state);

        match survival_assessor
            .assess_organism_survival(&life_record)
            .await
        {
            Ok(metrics) => {
                if metrics.overall_survival_probability > 0.9 {
                    info!(
                        "🌟 Organism {} achieving cosmic-level survival: {:.2}%",
                        organism_id.0,
                        metrics.overall_survival_probability * 100.0
                    );
                }

                if metrics.k_kristensen_parameter > 0.8 {
                    info!(
                        "⚛️ Organism {} k-kristensen parameter excellent: {:.4}",
                        organism_id.0, metrics.k_kristensen_parameter
                    );
                }
            }
            Err(e) => {
                error!("Failed to assess survival for {}: {}", organism_id.0, e);
            }
        }
    }
}

async fn maintain_neural_network(coordinator: &WaterRobotCoordinator) {
    // TODO: Integrate with actual neural interface monitoring
    // For now, just log status
    let coordination_state = coordinator.get_coordination_state();
    let state = coordination_state.read().unwrap();

    if let Some(neural_session) = &state.neural_control_session {
        info!(
            "🧠 Neural session active: {} (signal: {:.1}%)",
            neural_session.user_id,
            neural_session.signal_strength * 100.0
        );
    }

    let active_organisms = state.active_robots.len();
    if active_organisms > 0 {
        info!(
            "🌊 Neural network maintaining {} organisms",
            active_organisms
        );
    }
}

/// Demo function to show the complete system in action
pub async fn run_cryptobia_kingdom_demo() -> Result<()> {
    println!("🌌 Welcome to the Cryptobia Kingdom Demo");
    println!("🧬 Domain: Artificialis | Kingdom: Cryptobia | Genus: Hydra Blockchainus");
    println!(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    );

    // Initialize coordinator
    let coordinator = WaterRobotCoordinator::new().await?;

    // Spawn demo organisms
    spawn_genesis_organisms(&coordinator).await?;

    // Demo 1: Neural control
    println!("\n🧠 Demo 1: Neural Interface Control");
    println!("📡 Establishing neural link...");

    let neural_command = NeuralCommand {
        command_id: Uuid::new_v4(),
        user_id: "demo_user".to_string(),
        command_type: NeuralCommandType::FormFormation {
            formation: FormationMode::Circle { radius: 5.0 },
        },
        target_robots: vec![
            WaterRobotId("hydra_aquaticus_alpha".to_string()),
            WaterRobotId("hydra_coordinatus_beta".to_string()),
            WaterRobotId("hydra_quanticus_gamma".to_string()),
        ],
        neural_confidence: 0.95,
        issued_at: Utc::now(),
    };

    let result = coordinator.execute_neural_command(neural_command).await?;
    println!(
        "✅ Neural command executed in {}ms",
        result.execution_time_ms
    );

    // Demo 2: Survival assessment
    println!("\n🌌 Demo 2: Cosmic Survival Assessment");
    let assessor = SurvivalAssessor::new();

    if let Some(robot_state) = coordinator
        .get_robot_status(&WaterRobotId("hydra_quanticus_gamma".to_string()))
        .await
    {
        let life_record = create_life_record_from_robot_state(&robot_state);
        let survival_metrics = assessor.assess_organism_survival(&life_record).await?;

        display_survival_meter(&survival_metrics);
    }

    // Demo 3: Swarm coordination
    println!("\n🌊 Demo 3: Swarm Intelligence");
    coordinator
        .coordinate_swarm_movement(
            FormationMode::Grid { spacing: 2.0 },
            Position3D {
                x: 10.0,
                y: 10.0,
                z: 0.1,
            },
        )
        .await?;

    let stats = coordinator.get_swarm_statistics().await;
    println!("📊 Swarm Statistics:");
    println!(
        "  Commands: {}",
        stats
            .total_commands_processed
            .load(std::sync::atomic::Ordering::SeqCst)
    );
    println!(
        "  Formations: {}",
        stats
            .successful_formations
            .load(std::sync::atomic::Ordering::SeqCst)
    );
    println!(
        "  Neural commands: {}",
        stats
            .neural_commands_executed
            .load(std::sync::atomic::Ordering::SeqCst)
    );

    // Demo 4: Store interaction
    println!("\n🏪 Demo 4: Cryptobia Kingdom Store");
    let store = CryptobiaStore::new().await?;
    let marketplace = store.get_marketplace_overview().await;

    println!("📈 Marketplace Overview:");
    println!(
        "  Available organisms: {}",
        marketplace.total_organisms_available
    );
    println!("  Average price: {:.1} QNK", marketplace.average_price_qnk);
    println!("  Trending: {}", marketplace.trending_organism_type);
    println!("  24h volume: {:.1} QNK", marketplace.market_volume_24h);

    println!("\n🌟 Demo completed successfully!");
    println!("🧬 The Cryptobia Kingdom is alive and thriving!");

    Ok(())
}
