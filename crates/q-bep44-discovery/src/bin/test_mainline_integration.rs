/*!
# Q-NarwhalKnight Mainline DHT Integration Test

This tests the REAL BitTorrent DHT operations using the production mainline crate.
*/

use anyhow::Result;
use ed25519_dalek::SigningKey;
use q_bep44_discovery::real_discovery_engine::RealDiscoveryEngine;
use tracing::{info, error};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("\n🚀 Q-NARWHALKNIGHT MAINLINE DHT INTEGRATION TEST\n");
    println!("{}", "=".repeat(60));

    // Test 1: Create Real Discovery Engine with mainline DHT
    test_real_discovery_engine_creation().await?;

    println!("\n{}", "=".repeat(60));

    // Test 2: Test BEP-44 mutable data operations
    test_bep44_mutable_data_operations().await?;

    println!("\n{}", "=".repeat(60));
    println!("✅ ALL TESTS COMPLETED SUCCESSFULLY!");
    println!("🌐 Q-NarwhalKnight now has REAL BitTorrent DHT integration!");

    Ok(())
}

/// Test creating RealDiscoveryEngine with production mainline DHT
async fn test_real_discovery_engine_creation() -> Result<()> {
    println!("\n🧪 TEST 1: Real Discovery Engine Creation\n");

    let node_id = [42u8; 32]; // Test node ID

    info!("Creating RealDiscoveryEngine with production mainline DHT...");

    match RealDiscoveryEngine::new(node_id).await {
        Ok(mut engine) => {
            println!("✅ RealDiscoveryEngine created successfully!");

            // Test initialization
            match engine.initialize().await {
                Ok(_) => {
                    println!("✅ Discovery engine initialized successfully!");
                    println!("   • Mainline DHT: ACTIVE");
                    println!("   • BitTorrent network: CONNECTED");

                    // Test getting stats
                    let stats = engine.get_stats().await;
                    println!("📊 Discovery Stats:");
                    println!("   • Bootstrap attempts: {}", stats.bootstrap_attempts);
                    println!("   • Successful operations: {}", stats.successful_operations);

                    Ok(())
                }
                Err(e) => {
                    error!("❌ Failed to initialize discovery engine: {}", e);
                    Err(e)
                }
            }
        }
        Err(e) => {
            error!("❌ Failed to create RealDiscoveryEngine: {}", e);
            Err(e)
        }
    }
}

/// Test BEP-44 mutable data operations using real mainline DHT
async fn test_bep44_mutable_data_operations() -> Result<()> {
    println!("\n🧪 TEST 2: BEP-44 Mutable Data Operations\n");

    let node_id = [123u8; 32]; // Test node ID
    let mut engine = RealDiscoveryEngine::new(node_id).await?;
    engine.initialize().await?;

    // Test announcing presence
    let onion_address = "test-validator.onion";
    let capabilities = vec!["consensus".to_string(), "quantum".to_string()];

    println!("📡 Testing presence announcement...");
    match engine.announce_presence(onion_address, capabilities.clone()).await {
        Ok(_) => {
            println!("✅ Presence announced successfully!");
            println!("   • Onion: {}", onion_address);
            println!("   • Capabilities: {:?}", capabilities);

            // Wait a moment for propagation
            tokio::time::sleep(Duration::from_secs(2)).await;

            // Test discovering validators
            println!("\n🔍 Testing validator discovery...");
            match engine.discover_validators().await {
                Ok(validators) => {
                    println!("✅ Validator discovery completed!");
                    println!("   • Found {} validators", validators.len());

                    for (i, validator) in validators.iter().enumerate().take(3) {
                        println!("   • Validator {}: {} ({})",
                                i + 1,
                                validator.onion_address,
                                hex::encode(&validator.node_id[..8]));
                    }

                    Ok(())
                }
                Err(e) => {
                    error!("❌ Validator discovery failed: {}", e);
                    Err(e)
                }
            }
        }
        Err(e) => {
            error!("❌ Presence announcement failed: {}", e);
            Err(e)
        }
    }
}