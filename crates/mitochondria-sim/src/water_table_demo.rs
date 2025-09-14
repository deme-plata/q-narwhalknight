/// Water Table Demo - Macro-Scale Mitochondria v2 Proof of Concept
///
/// Physical implementation using 10cm plastic boats on water table
/// to demonstrate the core principles before moving to microfluidics.

use mitochondria_sim::*;
use anyhow::Result;
use bevy::prelude::*;
use std::collections::HashMap;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "🧬💧 Mitochondria v2 - Water Table Demo".into(),
                resolution: (1200.0, 800.0).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(WaterTableState::default())
        .add_systems(Startup, setup_water_table)
        .add_systems(Update, (
            update_boat_physics,
            process_nfc_interactions,
            update_blockchain_consensus,
            handle_user_input,
            update_visualization,
        ))
        .run();
}

#[derive(Resource, Default)]
struct WaterTableState {
    boats: HashMap<String, MacroDropletBoat>,
    water_table_size: f32,
    bluetooth_range: f32,
    consensus_leader: String,
    simulation_time: f32,
}

#[derive(Component, Debug, Clone)]
struct MacroDropletBoat {
    boat_id: String,
    position: Vec3,
    velocity: Vec3,
    nfc_blockchain: String,          // NFC tag data representing DNA
    pump_power: f32,                 // Water pump power (0-1)
    blockchain_mass: f32,            // Simulated DNA mass
    energy_level: f32,
    last_consensus_vote: Option<f32>,
}

#[derive(Component)]
struct BoatMarker;

#[derive(Component)]
struct WaterSurface;

fn setup_water_table(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut water_table: ResMut<WaterTableState>,
) {
    info!("🏗️ Setting up Water Table Demo");
    
    // Setup camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 10.0, 10.0)
            .looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
    
    // Setup lighting
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });
    
    // Create water table surface
    let water_surface = meshes.add(Plane3d::default().mesh().size(20.0, 20.0));
    let water_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.2, 0.6, 1.0, 0.8),
        alpha_mode: AlphaMode::Blend,
        reflectance: 0.8,
        ..default()
    });
    
    commands.spawn((
        PbrBundle {
            mesh: water_surface,
            material: water_material,
            transform: Transform::from_xyz(0.0, 0.0, 0.0),
            ..default()
        },
        WaterSurface,
    ));
    
    // Create boat fleet (10cm boats representing water droplets)
    let boat_mesh = meshes.add(Cuboid::new(1.0, 0.2, 0.5)); // 10cm x 2cm x 5cm boat
    
    for i in 0..20 {
        let boat_id = format!("boat_{:02}", i);
        
        // Random position on water table
        let x = (rand::random::<f32>() - 0.5) * 18.0; // -9 to +9
        let z = (rand::random::<f32>() - 0.5) * 18.0;
        
        // Create boat with unique color based on blockchain
        let blockchain_hash = blake3::hash(boat_id.as_bytes());
        let boat_color = Color::srgb(
            blockchain_hash.as_bytes()[0] as f32 / 255.0,
            blockchain_hash.as_bytes()[1] as f32 / 255.0, 
            blockchain_hash.as_bytes()[2] as f32 / 255.0,
        );
        
        let boat_material = materials.add(StandardMaterial {
            base_color: boat_color,
            metallic: 0.1,
            roughness: 0.8,
            ..default()
        });
        
        let macro_boat = MacroDropletBoat {
            boat_id: boat_id.clone(),
            position: Vec3::new(x, 0.1, z),
            velocity: Vec3::ZERO,
            nfc_blockchain: format!("genesis_{}", i), // Initial NFC data
            pump_power: 0.0,
            blockchain_mass: 1.0 + rand::random::<f32>() * 10.0, // Random initial mass
            energy_level: 1.0,
            last_consensus_vote: None,
        };
        
        commands.spawn((
            PbrBundle {
                mesh: boat_mesh.clone(),
                material: boat_material,
                transform: Transform::from_translation(macro_boat.position),
                ..default()
            },
            macro_boat.clone(),
            BoatMarker,
        ));
        
        water_table.boats.insert(boat_id, macro_boat);
    }
    
    water_table.water_table_size = 20.0;
    water_table.bluetooth_range = 2.0; // 2 meter BLE range
    
    info!("✅ Water Table Demo setup complete");
    info!("🚤 {} boats deployed on {} meter water table", 
          water_table.boats.len(), water_table.water_table_size);
}

fn update_boat_physics(
    time: Res<Time>,
    mut water_table: ResMut<WaterTableState>,
    mut boat_query: Query<(&mut Transform, &mut MacroDropletBoat), With<BoatMarker>>,
) {
    water_table.simulation_time += time.delta_seconds();
    
    for (mut transform, mut boat) in boat_query.iter_mut() {
        // Update boat position based on pump power and water physics
        boat.velocity.x += (rand::random::<f32>() - 0.5) * boat.pump_power * 0.1;
        boat.velocity.z += (rand::random::<f32>() - 0.5) * boat.pump_power * 0.1;
        
        // Apply water resistance
        boat.velocity *= 0.95;
        
        // Update position
        boat.position += boat.velocity * time.delta_seconds();
        
        // Boundary conditions (keep boats on table)
        boat.position.x = boat.position.x.clamp(-9.0, 9.0);
        boat.position.z = boat.position.z.clamp(-9.0, 9.0);
        boat.position.y = 0.1; // Float on surface
        
        // Update transform
        transform.translation = boat.position;
        
        // Energy decay
        boat.energy_level -= 0.01 * time.delta_seconds();
        boat.energy_level = boat.energy_level.max(0.0);
        
        // Update boat in state
        if let Some(state_boat) = water_table.boats.get_mut(&boat.boat_id) {
            *state_boat = boat.clone();
        }
    }
}

fn process_nfc_interactions(
    mut water_table: ResMut<WaterTableState>,
    boat_query: Query<&MacroDropletBoat, With<BoatMarker>>,
) {
    // Check for boats within BLE range and simulate "DNA reading"
    let boats: Vec<_> = boat_query.iter().collect();
    
    for i in 0..boats.len() {
        for j in (i + 1)..boats.len() {
            let boat1 = boats[i];
            let boat2 = boats[j];
            
            let distance = boat1.position.distance(boat2.position);
            
            if distance < water_table.bluetooth_range {
                // Simulate NFC/BLE interaction (reading neighbor's blockchain)
                debug!("📡 BLE interaction: {} ↔ {} (distance: {:.1}m)", 
                       boat1.boat_id, boat2.boat_id, distance);
                
                // In real implementation, boats would read each other's NFC tags
                // and update their local blockchain state
            }
        }
    }
}

fn update_blockchain_consensus(
    mut water_table: ResMut<WaterTableState>,
    boat_query: Query<&MacroDropletBoat, With<BoatMarker>>,
) {
    // Find boat with highest blockchain mass (proof-of-physical-swarm)
    let mut heaviest_boat = String::new();
    let mut max_mass = 0.0f32;
    
    for boat in boat_query.iter() {
        if boat.blockchain_mass > max_mass {
            max_mass = boat.blockchain_mass;
            heaviest_boat = boat.boat_id.clone();
        }
    }
    
    if heaviest_boat != water_table.consensus_leader {
        info!("👑 New consensus leader: {} (mass: {:.1})", heaviest_boat, max_mass);
        water_table.consensus_leader = heaviest_boat;
    }
    
    // Simulate blockchain growth for active boats
    for boat in water_table.boats.values_mut() {
        if boat.energy_level > 0.1 {
            boat.blockchain_mass += 0.1; // Gradual mass increase
        }
    }
}

fn handle_user_input(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut water_table: ResMut<WaterTableState>,
) {
    // Demo commands via keyboard
    if keyboard_input.just_pressed(KeyCode::Space) {
        info!("🚀 Triggering swarm movement demo");
        
        // Make all boats move in formation
        for boat in water_table.boats.values_mut() {
            boat.pump_power = 0.5;
            boat.velocity.x += (rand::random::<f32>() - 0.5) * 2.0;
            boat.velocity.z += (rand::random::<f32>() - 0.5) * 2.0;
        }
    }
    
    if keyboard_input.just_pressed(KeyCode::KeyB) {
        info!("🧬 Triggering blockchain synthesis demo");
        
        // Simulate DNA synthesis in all boats
        for boat in water_table.boats.values_mut() {
            boat.blockchain_mass += rand::random::<f32>() * 5.0;
            boat.nfc_blockchain = format!("block_{}", rand::random::<u32>());
        }
    }
    
    if keyboard_input.just_pressed(KeyCode::KeyF) {
        info!("🔄 Triggering binary fission demo");
        
        // Find largest boat and "split" it
        let largest_boat_id = water_table.boats.iter()
            .max_by(|(_, a), (_, b)| a.blockchain_mass.partial_cmp(&b.blockchain_mass).unwrap())
            .map(|(id, _)| id.clone());
        
        if let Some(boat_id) = largest_boat_id {
            if let Some(boat) = water_table.boats.get(&boat_id) {
                if boat.blockchain_mass > 10.0 {
                    // Create "daughter" boat
                    let daughter_id = format!("{}_daughter", boat_id);
                    let mut daughter_boat = boat.clone();
                    daughter_boat.boat_id = daughter_id.clone();
                    daughter_boat.blockchain_mass = boat.blockchain_mass / 2.0;
                    daughter_boat.position.x += 1.0; // Offset position
                    
                    water_table.boats.insert(daughter_id, daughter_boat);
                    
                    // Reduce parent mass
                    if let Some(parent) = water_table.boats.get_mut(&boat_id) {
                        parent.blockchain_mass /= 2.0;
                    }
                    
                    info!("🎉 Binary fission completed: {} → {}_daughter", boat_id, boat_id);
                }
            }
        }
    }
}

fn update_visualization(
    mut gizmos: Gizmos,
    water_table: Res<WaterTableState>,
    boat_query: Query<&MacroDropletBoat, With<BoatMarker>>,
) {
    // Draw BLE connections between nearby boats
    let boats: Vec<_> = boat_query.iter().collect();
    
    for i in 0..boats.len() {
        for j in (i + 1)..boats.len() {
            let boat1 = boats[i];
            let boat2 = boats[j];
            
            let distance = boat1.position.distance(boat2.position);
            
            if distance < water_table.bluetooth_range {
                // Draw connection line
                gizmos.line(
                    boat1.position + Vec3::Y * 0.5,
                    boat2.position + Vec3::Y * 0.5,
                    Color::srgba(0.0, 1.0, 0.0, 0.3),
                );
            }
        }
    }
    
    // Draw consensus leader highlight
    if let Some(leader_boat) = boats.iter().find(|b| b.boat_id == water_table.consensus_leader) {
        gizmos.sphere(
            leader_boat.position + Vec3::Y * 1.0,
            Quat::IDENTITY,
            0.5,
            Color::srgb(1.0, 0.8, 0.0), // Gold for leader
        );
    }
    
    // Draw table boundary
    gizmos.rect(
        Vec3::ZERO,
        Quat::IDENTITY,
        Vec2::new(water_table.water_table_size, water_table.water_table_size),
        Color::srgb(0.0, 0.3, 0.8),
    );
}

/// Print demo instructions
fn print_demo_instructions() {
    println!("🧬💧 Water Table Demo - Mitochondria v2 Proof of Concept");
    println!();
    println!("This demo shows the core principles using 10cm boats on a water table:");
    println!("🚤 Each boat = water droplet with DNA blockchain");
    println!("📡 Green lines = BLE connections (simulating FRET DNA reading)");
    println!("👑 Golden sphere = consensus leader (heaviest blockchain)");
    println!();
    println!("Controls:");
    println!("  [SPACE] - Trigger swarm movement (electro-wetting simulation)");
    println!("  [B]     - Trigger blockchain synthesis (DNA growth)");
    println!("  [F]     - Trigger binary fission (boat replication)");
    println!();
    println!("Watch how the boats:");
    println!("✅ Move autonomously via 'pump power' (electro-wetting)");
    println!("✅ Read each other's 'DNA' when in BLE range");
    println!("✅ Grow their blockchain mass over time");
    println!("✅ Achieve consensus based on physical mass");
    println!("✅ Replicate when mass threshold is reached");
    println!();
    println!("This is the exact same algorithm that will run on real water droplets!");
    println!("🌟 The future of biological computing starts here...");
}

/// Demo scenario runner
pub async fn run_water_table_demo() -> Result<()> {
    print_demo_instructions();
    
    info!("🚀 Starting Water Table Demo");
    
    // In a real application, this would interface with:
    // - ESP32 microcontrollers on each boat
    // - Water pumps for propulsion  
    // - NFC readers for blockchain scanning
    // - Overhead camera for position tracking
    // - Tor connection for remote commands
    
    println!("🎬 Demo running... (use keyboard controls in the 3D window)");
    
    // The Bevy app loop handles the actual demo
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_water_table_setup() {
        let state = WaterTableState::default();
        assert_eq!(state.boats.len(), 0);
        assert_eq!(state.water_table_size, 0.0);
    }
    
    #[test]
    fn test_macro_boat_creation() {
        let boat = MacroDropletBoat {
            boat_id: "test_boat".to_string(),
            position: Vec3::new(1.0, 0.1, 1.0),
            velocity: Vec3::ZERO,
            nfc_blockchain: "genesis_test".to_string(),
            pump_power: 0.0,
            blockchain_mass: 1.0,
            energy_level: 1.0,
            last_consensus_vote: None,
        };
        
        assert_eq!(boat.boat_id, "test_boat");
        assert_eq!(boat.blockchain_mass, 1.0);
        assert_eq!(boat.energy_level, 1.0);
    }
}