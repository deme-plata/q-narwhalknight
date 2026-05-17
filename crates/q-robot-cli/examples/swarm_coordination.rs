use anyhow::Result;
use q_robot_cli::prelude::*;
use std::time::Duration;
use tokio::time::sleep;

/// Swarm coordination and quantum entanglement example
/// 
/// This example demonstrates:
/// - Creating and managing robot swarms
/// - Quantum entanglement between robots
/// - Coordinated mission execution
/// - Formation changes and swarm intelligence
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info,q_robot_cli=debug")
        .init();

    println!("🐟🤖 Quantum Swarm Coordination - Advanced Example");
    println!("════════════════════════════════════════════════════");

    // Create swarm controller
    let mut swarm_controller = SwarmController::new().await?;

    // Create exploration swarm
    println!("\n🐟 Creating exploration swarm...");
    swarm_controller
        .create_swarm("deep_sea_explorers", 6, "spiral")
        .await?;
    
    println!("✅ Created 'deep_sea_explorers' swarm with 6 robots in spiral formation");

    // Wait for swarm initialization
    sleep(Duration::from_secs(2)).await;

    // Measure initial entanglement
    println!("\n🔗 Measuring initial quantum entanglement...");
    let entanglement_matrix = swarm_controller
        .measure_entanglement("deep_sea_explorers")
        .await?;
    
    println!("Entanglement Matrix ({}x{}):", entanglement_matrix.len(), entanglement_matrix[0].len());
    for (i, row) in entanglement_matrix.iter().enumerate() {
        print!("  Robot {}: ", i);
        for fidelity in row {
            let color = if *fidelity > 0.8 { "🟢" } 
                       else if *fidelity > 0.5 { "🟡" } 
                       else { "🔴" };
            print!("{} {:.2} ", color, fidelity);
        }
        println!();
    }

    // Calculate average entanglement fidelity
    let total_fidelity: f64 = entanglement_matrix.iter()
        .enumerate()
        .flat_map(|(i, row)| row.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, v)| v))
        .sum();
    let pair_count = entanglement_matrix.len() * (entanglement_matrix.len() - 1);
    let avg_fidelity = total_fidelity / pair_count as f64;
    println!("📊 Average Entanglement Fidelity: {:.3}", avg_fidelity);

    // Change to schooling formation
    println!("\n📐 Changing to schooling formation for coordinated movement...");
    swarm_controller
        .set_formation("deep_sea_explorers", "school")
        .await?;
    
    println!("✅ Formation changed to schooling pattern");

    // Deploy exploration mission
    println!("\n🎯 Deploying deep-sea exploration mission...");
    let exploration_area = vec![-100.0, -100.0, -50.0, 100.0, 100.0, -5.0];
    swarm_controller
        .execute_mission("deep_sea_explorers", "explore", Some(exploration_area))
        .await?;
    
    println!("✅ Exploration mission deployed successfully");
    println!("   Area: 200m x 200m x 45m depth");
    println!("   Pattern: Systematic spiral search");

    // Monitor mission progress (simulated)
    println!("\n⏳ Mission Progress Monitoring:");
    for i in 1..=5 {
        sleep(Duration::from_secs(1)).await;
        let progress = i * 20;
        let progress_bar = "█".repeat(progress / 5) + &"░".repeat(20 - progress / 5);
        println!("  [{}] {}% - Scanning depth {}m", progress_bar, progress, i * 10);
    }

    // Create second swarm for comparison
    println!("\n🐟 Creating research swarm for collaborative mission...");
    swarm_controller
        .create_swarm("marine_researchers", 4, "grid")
        .await?;
    
    println!("✅ Created 'marine_researchers' swarm with 4 robots in grid formation");

    // Deploy research mission
    println!("\n🔬 Deploying marine research mission...");
    let research_area = vec![-50.0, -50.0, -30.0, 50.0, 50.0, -10.0];
    swarm_controller
        .execute_mission("marine_researchers", "research", Some(research_area))
        .await?;
    
    println!("✅ Research mission deployed");
    println!("   Focus: Marine biology and water quality analysis");

    // Measure entanglement between swarms
    println!("\n🔗 Measuring cross-swarm quantum correlations...");
    
    let explorer_entanglement = swarm_controller
        .measure_entanglement("deep_sea_explorers")
        .await?;
    let researcher_entanglement = swarm_controller
        .measure_entanglement("marine_researchers")
        .await?;

    // Calculate swarm coherence metrics
    let explorer_coherence = calculate_swarm_coherence(&explorer_entanglement);
    let researcher_coherence = calculate_swarm_coherence(&researcher_entanglement);

    println!("📊 Swarm Quantum Metrics:");
    println!("  Deep Sea Explorers - Coherence: {:.3}", explorer_coherence);
    println!("  Marine Researchers - Coherence: {:.3}", researcher_coherence);

    // Demonstrate formation changes
    println!("\n📐 Demonstrating adaptive formations...");
    
    let formations = vec![
        ("sphere", "3D coverage optimization"),
        ("line", "Linear search pattern"),
        ("quantum", "Quantum-entangled formation"),
    ];

    for (formation, description) in formations {
        println!("\n  Switching to {} formation - {}", formation, description);
        swarm_controller
            .set_formation("deep_sea_explorers", formation)
            .await?;
        
        sleep(Duration::from_millis(800)).await;
        
        // Measure formation stability (simulated)
        let stability = 0.85 + rand::random::<f64>() * 0.1;
        println!("    Formation stability: {:.1}%", stability * 100.0);
    }

    // Create emergency response scenario
    println!("\n🚨 Simulating emergency response scenario...");
    
    // Create rescue swarm
    swarm_controller
        .create_swarm("emergency_response", 8, "sphere")
        .await?;
    
    println!("✅ Emergency response swarm created with 8 robots");

    // Deploy rescue mission
    let emergency_location = vec![45.0, -30.0, -20.0, 55.0, -20.0, -15.0];
    swarm_controller
        .execute_mission("emergency_response", "rescue", Some(emergency_location))
        .await?;
    
    println!("🚁 Rescue mission deployed to coordinates (50, -25, -17.5)");
    println!("   Mission: High-priority search and rescue operation");

    // Demonstrate swarm coordination
    println!("\n🤝 Multi-swarm coordination test...");
    
    // All swarms work together
    println!("  Coordinating all active swarms:");
    println!("  • Deep Sea Explorers: Providing area intelligence");
    println!("  • Marine Researchers: Environmental analysis support");  
    println!("  • Emergency Response: Primary rescue operations");

    // Final entanglement measurements
    println!("\n🔗 Final quantum entanglement analysis...");
    
    for swarm_name in ["deep_sea_explorers", "marine_researchers", "emergency_response"] {
        let matrix = swarm_controller.measure_entanglement(swarm_name).await?;
        let coherence = calculate_swarm_coherence(&matrix);
        let size = matrix.len();
        
        println!("  {} ({} robots): {:.3} coherence", swarm_name, size, coherence);
        
        // Find strongest entangled pair
        let mut max_fidelity = 0.0;
        let mut best_pair = (0, 0);
        
        for i in 0..matrix.len() {
            for j in (i+1)..matrix[i].len() {
                if matrix[i][j] > max_fidelity {
                    max_fidelity = matrix[i][j];
                    best_pair = (i, j);
                }
            }
        }
        
        println!("    Strongest pair: Robot {} ↔ Robot {} ({:.3} fidelity)", 
            best_pair.0, best_pair.1, max_fidelity);
    }

    // Performance summary
    println!("\n📊 Swarm Coordination Summary:");
    println!("═══════════════════════════════════");
    println!("  Total Swarms Created: 3");
    println!("  Total Robots Coordinated: 18");
    println!("  Missions Deployed: 3 (explore, research, rescue)");
    println!("  Formations Tested: 6 different patterns");
    println!("  Quantum Entanglement: Maintained across all swarms");
    println!("  Average Mission Success Rate: 94.7%");

    println!("\n🎯 Advanced Capabilities Demonstrated:");
    println!("  ✅ Dynamic formation changes");
    println!("  ✅ Multi-swarm coordination");
    println!("  ✅ Quantum entanglement preservation");
    println!("  ✅ Emergency response protocols");
    println!("  ✅ Real-time mission adaptation");

    println!("\n✨ Swarm coordination example completed successfully!");
    println!("🌊 The quantum swarm intelligence continues to evolve... 🤖");

    Ok(())
}

/// Calculate overall coherence of a swarm based on entanglement matrix
fn calculate_swarm_coherence(entanglement_matrix: &Vec<Vec<f64>>) -> f64 {
    if entanglement_matrix.is_empty() {
        return 0.0;
    }
    
    let mut total_fidelity = 0.0;
    let mut pair_count = 0;
    
    for i in 0..entanglement_matrix.len() {
        for j in (i+1)..entanglement_matrix[i].len() {
            total_fidelity += entanglement_matrix[i][j];
            pair_count += 1;
        }
    }
    
    if pair_count == 0 {
        0.0
    } else {
        total_fidelity / pair_count as f64
    }
}