/// Integration tests for distributed AI system
///
/// Tests the Mistral.rs distributed AI compute functionality
use anyhow::Result;
use q_robot_control::distributed_ai::{
    demo_distributed_ai_compute, demo_gguf_splitting_with_economy,
};

#[tokio::test]
async fn test_basic_distributed_ai_compute() -> Result<()> {
    println!("🧠 Testing basic distributed AI compute demo");

    // Run the basic demo function
    match demo_distributed_ai_compute().await {
        Ok(_) => {
            println!("✅ Basic distributed AI compute demo completed successfully");
            Ok(())
        }
        Err(e) => {
            println!("❌ Basic demo failed: {}", e);
            Err(e)
        }
    }
}

#[tokio::test]
async fn test_gguf_splitting_with_economy() -> Result<()> {
    println!("💰 Testing GGUF model splitting with token economy");

    // Run the token economy demo function
    match demo_gguf_splitting_with_economy().await {
        Ok(_) => {
            println!("✅ GGUF splitting with token economy demo completed successfully");
            Ok(())
        }
        Err(e) => {
            println!("❌ Token economy demo failed: {}", e);
            Err(e)
        }
    }
}
