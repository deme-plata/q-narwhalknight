//! Example: Ethereum transaction privacy with Q-NarwhalKnight PaaS

use q_paas_sdk::{PaaSClient, EthereumWallet, PrivacyLevel};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init();

    // Step 1: Initialize PaaS client
    let api_key = std::env::var("Q_PAAS_API_KEY")
        .expect("Q_PAAS_API_KEY environment variable not set");
    
    let client = PaaSClient::new(
        api_key,
        "https://quillon.xyz".to_string()
    );

    println!("✓ PaaS client initialized");

    // Step 2: Create Ethereum wallet
    let private_key = std::env::var("ETHEREUM_PRIVATE_KEY")
        .expect("ETHEREUM_PRIVATE_KEY environment variable not set");
    
    let wallet = EthereumWallet::from_private_key(&private_key)?;
    println!("✓ Ethereum wallet loaded");
    println!("  Address: {:?}", wallet.address());

    // Step 3: Check API key info
    let key_info = client.get_api_key_info().await?;
    println!("✓ API Key Info:");
    println!("  Tier: {:?}", key_info.tier);
    println!("  Rate limit: {}/min", key_info.rate_limit);

    // Step 4: Mix a transaction with MEV protection
    println!("\n🔄 Mixing Ethereum transaction with MEV protection...");
    
    let signed_tx_hex = "0xf86c..."; // Your signed transaction here
    
    let result = client.mix_ethereum_transaction(
        signed_tx_hex,
        PrivacyLevel::Enhanced,
        true // MEV protection
    ).await?;

    println!("✅ Transaction mixed successfully!");
    println!("  Mix ID: {}", result.mix_id);
    println!("  Status: {:?}", result.status);
    println!("  ETA: {} seconds", result.estimated_completion_seconds);

    Ok(())
}
