use anyhow::Result;
use q_robot_cli::prelude::*;
use std::time::Duration;
use tokio::time::sleep;

/// Basic robot control example
/// 
/// This example demonstrates:
/// - Connecting to robots
/// - Basic movement and control
/// - Sensor data reading
/// - Quantum ability activation
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info,q_robot_cli=debug")
        .init();

    println!("🌊🤖 Quantum Water Robot Control - Basic Example");
    println!("═══════════════════════════════════════════════════");

    // Load configuration
    let config = RobotConfig::create_example();
    let mut robot_manager = RobotManager::new(config).await?;

    // Connect to a quantum jellyfish robot
    let robot_id = RobotId::new("quantum_jelly_001");
    println!("\n🔌 Connecting to robot: {}", robot_id);
    
    let connected = robot_manager
        .connect_robot(robot_id.clone(), Some("jellyfish".to_string()))
        .await?;
    
    if connected {
        println!("✅ Successfully connected to robot!");
    } else {
        println!("❌ Failed to connect to robot");
        return Ok(());
    }

    // Get initial robot status
    println!("\n📊 Initial Robot Status:");
    let status = robot_manager.get_robot_status("quantum_jelly_001").await?;
    println!("  Position: ({:.2}, {:.2}, {:.2})", 
        status.position.0, status.position.1, status.position.2);
    println!("  Battery: {:.1}%", status.battery_level);
    println!("  Quantum Coherence: {:.3} ms", status.quantum_coherence * 1000.0);

    // Move robot to a new position
    println!("\n🎯 Moving robot to new coordinates...");
    let target_position = vec![25.0, -15.0, -8.0];
    robot_manager
        .move_robot("quantum_jelly_001", target_position.clone(), 0.6)
        .await?;
    
    // Wait for movement to complete
    sleep(Duration::from_secs(2)).await;
    
    // Check new position
    let updated_status = robot_manager.get_robot_status("quantum_jelly_001").await?;
    println!("  New Position: ({:.2}, {:.2}, {:.2})", 
        updated_status.position.0, updated_status.position.1, updated_status.position.2);

    // Activate bioluminescence ability
    println!("\n⚡ Activating bioluminescence ability...");
    robot_manager
        .activate_ability("quantum_jelly_001", "bioluminescence", vec!["0.8".to_string()])
        .await?;
    
    // Check sensor data
    println!("\n📊 Current Sensor Readings:");
    let final_status = robot_manager.get_robot_status("quantum_jelly_001").await?;
    let sensors = &final_status.sensor_data;
    
    println!("  Water Temperature: {:.1}°C", sensors.temperature);
    println!("  pH Level: {:.2}", sensors.ph);
    println!("  Dissolved Oxygen: {:.1} mg/L", sensors.dissolved_oxygen);
    println!("  Pressure: {:.0} Pa", sensors.pressure);
    println!("  Salinity: {:.1} PSU", sensors.salinity);
    println!("  Bioluminescence: {:.0} lumens", sensors.bioluminescence_intensity);

    // Perform environmental scan
    println!("\n🌊 Performing environmental scan...");
    let scan_results = robot_manager
        .scan_environment(50.0, 25.0)
        .await?;
    
    println!("  Temperature: {:.1}°C", scan_results.temperature);
    println!("  Species Detected: {}", scan_results.species_count);
    println!("  Coral Health: {:.1}%", scan_results.coral_health);
    println!("  Pollution Level: {}", scan_results.pollution_level);

    // Check water quality
    println!("\n💧 Water Quality Assessment:");
    let water_quality = robot_manager.check_water_quality().await?;
    println!("  Overall Rating: {}", water_quality.overall_rating);
    println!("  pH: {:.2}", water_quality.ph);
    println!("  Dissolved Oxygen: {:.1} mg/L", water_quality.dissolved_oxygen);
    println!("  Turbidity: {:.1} NTU", water_quality.turbidity);

    // Track marine life
    println!("\n🐠 Tracking Marine Life:");
    let marine_life = robot_manager.track_marine_life(None).await?;
    
    if marine_life.is_empty() {
        println!("  No marine life detected in current area");
    } else {
        for life in marine_life.iter().take(3) {
            println!("  {} - {} individuals at ({:.1}, {:.1}, {:.1})",
                life.species, life.count, life.location.0, life.location.1, life.location.2);
        }
    }

    // Demonstrate quantum superposition ability (if available)
    if let Ok(_) = robot_manager
        .activate_ability("quantum_jelly_001", "superposition_glow", vec![])
        .await 
    {
        println!("\n⚛️ Quantum superposition activated!");
        
        // Check quantum coherence after activation
        let quantum_status = robot_manager.get_robot_status("quantum_jelly_001").await?;
        println!("  Updated Quantum Coherence: {:.3} ms", 
            quantum_status.quantum_coherence * 1000.0);
    }

    // Final status report
    println!("\n📋 Final Robot Status:");
    let final_status = robot_manager.get_robot_status("quantum_jelly_001").await?;
    println!("  Active Abilities: {}", final_status.active_abilities.join(", "));
    println!("  Connection Quality: {:.1}%", final_status.connection_quality * 100.0);
    println!("  Battery Remaining: {:.1}%", final_status.battery_level);

    // List all connected robots
    println!("\n🤖 All Connected Robots:");
    let all_robots = robot_manager.list_robots().await?;
    for robot in all_robots {
        println!("  • {} ({}) - {} - {:.1}% battery", 
            robot.id, robot.robot_type, robot.status, robot.battery_level);
    }

    println!("\n✅ Basic robot control example completed successfully!");
    println!("🌊 The quantum ocean awaits your next command... 🤖");

    Ok(())
}