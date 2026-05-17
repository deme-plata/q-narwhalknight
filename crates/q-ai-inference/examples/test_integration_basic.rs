//! Basic integration test for distributed AI inference
//!
//! This example tests the MistralIntegration without requiring actual model weights.
//! It verifies that the privacy layer, KV-cache, pipeline, and load balancer
//! are correctly initialized and working together.

use q_ai_inference::MistralIntegration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🚀 Q-NarwhalKnight Distributed AI - Basic Integration Test\n");

    // Initialize the integration
    println!("📦 Initializing MistralIntegration with default config...");
    let mut integration = MistralIntegration::new().await?;
    println!("✅ Integration initialized successfully!\n");

    // Test with a simple prompt
    let prompt = "hello";
    println!("🔤 Testing with prompt: \"{}\"", prompt);
    println!("   (This will use placeholder tokenization/detokenization)\n");

    // Generate with privacy enabled
    println!("🛡️  Generating with privacy enabled...");
    let (response, stats) = integration.generate_with_privacy(
        prompt,
        true,  // enable_encryption
        true,  // enable_zk_proofs
    ).await?;

    println!("\n📊 Generation Results:");
    println!("   Response: \"{}\"", response);
    println!("   Tokens generated: {}", stats.tokens_generated);
    println!("   Generation time: {:.2}ms", stats.generation_time_ms);
    println!("   Privacy overhead: {:.2}ms", stats.privacy_overhead_ms);
    println!("   Distribution overhead: {:.2}ms", stats.distribution_overhead_ms);
    println!("   Total time: {:.2}ms", stats.total_time_ms);
    println!("   Tokens/sec: {:.2}", stats.tokens_per_second);

    // Get system statistics
    println!("\n📈 System Statistics:");
    let system_stats = integration.get_stats().await?;
    println!("{}", system_stats);

    println!("\n✅ Integration test completed successfully!");
    println!("🌟 Next step: Integrate actual GGUF model loading and tokenization");

    Ok(())
}
