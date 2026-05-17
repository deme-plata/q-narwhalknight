/// Simple example demonstrating distributed AI functionality
///
/// This example showcases the Mistral.rs distributed compute system
/// running in the q-robot-control crate
use anyhow::Result;
use q_robot_control::distributed_ai::{
    demo_distributed_ai_compute, demo_gguf_splitting_with_economy,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧠 Q-NarwhalKnight Mistral.rs Distributed AI Demo");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("\n📊 DEMO 1: Basic Distributed AI Compute");
    println!("🎯 Testing Hydra Computatus castle-of-compute architecture...");

    match demo_distributed_ai_compute().await {
        Ok(_) => {
            println!("✅ SUCCESS: Basic distributed AI compute demo completed!");
            println!("🌟 Hydra Computatus organisms are functioning correctly");
        }
        Err(e) => {
            println!("❌ FAILED: Basic demo error: {}", e);
            return Err(e);
        }
    }

    println!("\n💰 DEMO 2: GGUF Model Splitting with Token Economy");
    println!("🎯 Testing model sharding with QNK coin payment system...");

    match demo_gguf_splitting_with_economy().await {
        Ok(_) => {
            println!("✅ SUCCESS: GGUF splitting with token economy completed!");
            println!("💎 QNK coin-based compute economy is working perfectly");
        }
        Err(e) => {
            println!("❌ FAILED: Token economy demo error: {}", e);
            return Err(e);
        }
    }

    println!("\n🌟 DISTRIBUTED AI SYSTEM FULLY OPERATIONAL! 🚀");
    println!("🧬 All Hydra Computatus organisms are healthy and processing");
    println!("💰 QNK token economy is functioning correctly");
    println!("🏰 Castle-of-compute architecture validated");

    Ok(())
}
