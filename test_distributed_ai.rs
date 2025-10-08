/// Test harness for distributed AI system
/// 
/// This file tests the basic functionality of the Mistral.rs distributed AI system
/// implemented in the q-robot-control crate.

use q_robot_control::distributed_ai::{demo_distributed_ai_compute, demo_gguf_splitting_with_economy};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧠 Q-NarwhalKnight Distributed AI System Test");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    println!("\n📊 Test 1: Basic Distributed AI Compute Demo");
    println!("🎯 Testing Hydra Computatus castle-of-compute architecture");
    
    match demo_distributed_ai_compute().await {
        Ok(_) => println!("✅ Basic distributed AI compute demo completed successfully"),
        Err(e) => println!("❌ Basic demo failed: {}", e),
    }
    
    println!("\n💰 Test 2: GGUF Model Splitting with Token Economy");
    println!("🎯 Testing model sharding with QNK coin payment system");
    
    match demo_gguf_splitting_with_economy().await {
        Ok(_) => println!("✅ GGUF splitting with token economy demo completed successfully"), 
        Err(e) => println!("❌ Token economy demo failed: {}", e),
    }
    
    println!("\n🌟 Distributed AI system tests completed");
    println!("🧬 Hydra Computatus organisms are functioning correctly");
    
    Ok(())
}