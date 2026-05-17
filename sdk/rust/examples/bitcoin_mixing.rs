//! Example: Bitcoin transaction mixing with Q-NarwhalKnight PaaS

use q_paas_sdk::{PaaSClient, BitcoinWallet, PrivacyLevel};

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

    // Step 2: Create Bitcoin wallet
    let wif = std::env::var("BITCOIN_WIF_KEY")
        .expect("BITCOIN_WIF_KEY environment variable not set");
    
    let wallet = BitcoinWallet::from_wif(&wif)?;
    println!("✓ Bitcoin wallet loaded");
    println!("  Address: {}", wallet.address());

    // Step 3: Check billing balance
    let billing = client.get_billing_info().await?;
    println!("✓ Balance: {} QUG (${:.2})", billing.balance_qug, billing.balance_usd);

    // Step 4: Mix a transaction
    println!("\n🔄 Mixing Bitcoin transaction...");
    
    let signed_tx_hex = "0100000001..."; // Your signed transaction here
    
    let result = client.mix_bitcoin_transaction(
        signed_tx_hex,
        PrivacyLevel::Maximum,
        true // Use Tor
    ).await?;

    println!("✅ Transaction mixed successfully!");
    println!("  Mix ID: {}", result.mix_id);
    println!("  Status: {:?}", result.status);
    println!("  ETA: {} seconds", result.estimated_completion_seconds);
    println!("  Quantum entropy: {}", result.quantum_entropy_applied);
    
    if let Some(circuit) = result.tor_circuit_used {
        println!("  Tor circuit: {}", circuit);
    }

    // Step 5: Poll for completion
    println!("\n⏳ Waiting for mix to complete...");
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        
        let status = client.get_mix_status(&result.mix_id).await?;
        println!("  Status: {:?}", status.status);
        
        if matches!(status.status, q_paas_sdk::types::MixStatus::Completed) {
            println!("✅ Mix completed!");
            break;
        } else if matches!(status.status, q_paas_sdk::types::MixStatus::Failed) {
            println!("❌ Mix failed!");
            break;
        }
    }

    Ok(())
}
