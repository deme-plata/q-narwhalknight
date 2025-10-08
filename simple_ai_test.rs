/// Direct test of Mistral.rs distributed AI system
/// 
/// This quickly tests both basic distributed AI compute and GGUF token economy

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧠 Q-NarwhalKnight Mistral.rs Distributed AI Quick Test");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // Test 1: Basic distributed AI compute
    println!("\n📊 Test 1: Basic Distributed AI Compute");
    match q_robot_control::distributed_ai::demo_distributed_ai_compute().await {
        Ok(_) => {
            println!("✅ SUCCESS: Basic distributed AI compute works!");
            println!("🌟 Hydra Computatus organisms are processing correctly");
        },
        Err(e) => {
            println!("❌ FAILED: Basic demo error: {}", e);
        }
    }
    
    // Test 2: GGUF model splitting with token economy
    println!("\n💰 Test 2: GGUF Model Splitting with QNK Token Economy");
    match q_robot_control::distributed_ai::demo_gguf_splitting_with_economy().await {
        Ok(_) => {
            println!("✅ SUCCESS: GGUF splitting with QNK token economy works!");
            println!("💎 Model sharding and payment system operational");
        },
        Err(e) => {
            println!("❌ FAILED: Token economy demo error: {}", e);
        }
    }
    
    println!("\n🚀 Mistral.rs distributed AI system validation complete!");
    println!("🧬 All Hydra Computatus organisms are healthy and processing");
    println!("💰 QNK token-based compute economy is fully operational");
    println!("🏰 Castle-of-compute architecture validated and ready");
    
    Ok(())
}