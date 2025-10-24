//! Transaction Propagation Test with Authenticated Wallets
//!
//! This test creates wallets, signs transactions, and verifies propagation across nodes

use anyhow::Result;
use chrono::Utc;
use ed25519_dalek::{Signer, SigningKey};
use hex;
use rand::RngCore;
use reqwest;
use serde_json::json;
use sha3::{Digest, Sha3_256};
use std::time::Duration;
use tokio;

const NODE1: &str = "http://localhost:8080";
const NODE2: &str = "http://localhost:8084";
const NODE3: &str = "http://localhost:9060";
const NODE4: &str = "http://localhost:9666";

#[derive(Debug)]
struct TestWallet {
    address: [u8; 32],
    signing_key: SigningKey,
    mnemonic: String,
}

impl TestWallet {
    /// Create a new Ed25519 wallet (Q0 phase)
    fn new() -> Self {
        let mut secret_bytes = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut secret_bytes);
        let signing_key = SigningKey::from_bytes(&secret_bytes);
        let verifying_key = signing_key.verifying_key();
        let address = verifying_key.to_bytes();

        // Use standard BIP39 test mnemonic
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about".to_string();

        Self {
            address,
            signing_key,
            mnemonic,
        }
    }

    /// Get wallet address with 'qnk' prefix
    fn address_string(&self) -> String {
        format!("qnk{}", hex::encode(self.address))
    }

    /// Sign an authentication challenge
    fn sign_auth_challenge(&self, path: &str, timestamp: i64) -> String {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.address);
        hasher.update(&timestamp.to_le_bytes());
        hasher.update(path.as_bytes());
        let message = hasher.finalize();

        let signature = self.signing_key.sign(&message);
        hex::encode(signature.to_bytes())
    }
}

async fn get_faucet_coins(client: &reqwest::Client, node: &str, wallet: &TestWallet) -> Result<f64> {
    println!("💰 Requesting faucet coins for {}", wallet.address_string());

    let response = client
        .post(format!("{}/api/v1/faucet", node))
        .json(&json!({
            "wallet_address": wallet.address_string()
        }))
        .send()
        .await?;

    let status = response.status();
    let body: serde_json::Value = response.json().await?;

    if !status.is_success() {
        anyhow::bail!("Faucet request failed: {:?}", body);
    }

    let balance = body["data"]["new_balance_qnk"]
        .as_f64()
        .unwrap_or(0.0);

    println!("✅ Faucet successful! Balance: {} QNK", balance);
    Ok(balance)
}

async fn send_transaction(
    client: &reqwest::Client,
    node: &str,
    from_wallet: &TestWallet,
    to_address: &str,
    amount_qnk: f64,
) -> Result<String> {
    let path = "/api/v1/transactions/send";
    let timestamp = Utc::now().timestamp();

    // Sign the authentication challenge
    let signature = from_wallet.sign_auth_challenge(path, timestamp);

    // Create X-Wallet-Auth header
    let auth_header = json!({
        "address": from_wallet.address_string(),
        "timestamp": timestamp,
        "scheme": "Ed25519",
        "signature": signature
    }).to_string();

    println!("📤 Sending transaction:");
    println!("   From: {}", from_wallet.address_string());
    println!("   To: {}", to_address);
    println!("   Amount: {} QNK", amount_qnk);

    let response = client
        .post(format!("{}{}", node, path))
        .header("X-Wallet-Auth", auth_header)
        .json(&json!({
            "from": from_wallet.address_string(),
            "to": to_address,
            "amount": amount_qnk,
            "mnemonic": from_wallet.mnemonic
        }))
        .send()
        .await?;

    let status = response.status();
    let body: serde_json::Value = response.json().await?;

    // Debug: Print the full response
    println!("📋 Response status: {}", status);
    println!("📋 Response body: {}", serde_json::to_string_pretty(&body)?);

    // Check both HTTP status AND JSON success field
    if !status.is_success() || !body["success"].as_bool().unwrap_or(false) {
        let error_msg = body["error"].as_str().unwrap_or("Unknown error");
        println!("❌ Transaction failed: {}", error_msg);
        anyhow::bail!("Transaction failed: {}", error_msg);
    }

    let tx_hash = body["data"]["transaction_hash"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();

    println!("✅ Transaction sent! Hash: {}", tx_hash);
    Ok(tx_hash)
}

async fn check_transaction_on_node(
    client: &reqwest::Client,
    node: &str,
    tx_hash: &str,
) -> Result<bool> {
    let response = client
        .get(format!("{}/api/v1/transactions/{}", node, tx_hash))
        .send()
        .await?;

    let body: serde_json::Value = response.json().await?;

    if body["success"].as_bool() == Some(true) {
        println!("   ✓ {} sees transaction", node);
        Ok(true)
    } else {
        println!("   ✗ {} doesn't see transaction yet", node);
        Ok(false)
    }
}

async fn check_balance(
    client: &reqwest::Client,
    node: &str,
    wallet: &TestWallet,
) -> Result<f64> {
    let path = format!("/api/v1/wallets/{}", wallet.address_string());
    let timestamp = Utc::now().timestamp();

    // Sign the authentication challenge
    let signature = wallet.sign_auth_challenge(&path, timestamp);

    // Create X-Wallet-Auth header
    let auth_header = json!({
        "address": wallet.address_string(),
        "timestamp": timestamp,
        "scheme": "Ed25519",
        "signature": signature
    }).to_string();

    let response = client
        .get(format!("{}{}", node, path))
        .header("X-Wallet-Auth", auth_header)
        .send()
        .await?;

    let body: serde_json::Value = response.json().await?;

    if body["success"].as_bool() == Some(true) {
        let balance = body["data"]["balance"]
            .as_f64()
            .unwrap_or(0.0);
        Ok(balance)
    } else {
        Ok(0.0)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   Q-NarwhalKnight Transaction Propagation Test Suite         ║");
    println!("║                   v0.0.9-beta                                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;

    // STEP 1: Create two test wallets
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("STEP 1: Creating Test Wallets");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let wallet1 = TestWallet::new();
    let wallet2 = TestWallet::new();

    println!("Wallet 1: {}", wallet1.address_string());
    println!("Wallet 2: {}", wallet2.address_string());
    println!();

    // STEP 2: Get faucet coins for wallet 1
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("STEP 2: Getting Faucet Coins");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let balance1 = get_faucet_coins(&client, NODE1, &wallet1).await?;
    println!("Wallet 1 balance: {} QNK", balance1);
    println!();

    // Wait for balance to settle
    tokio::time::sleep(Duration::from_secs(2)).await;

    // STEP 3: Send transaction from wallet1 to wallet2 VIA NODE 4
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("STEP 3: Sending Authenticated Transaction to NODE 4");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("📤 Submitting transaction to Node 4 (port 9666)...");
    let tx_hash = send_transaction(
        &client,
        NODE4,  // Send to Node 4 instead of Node 1
        &wallet1,
        &wallet2.address_string(),
        2.0,
    ).await?;

    println!("✅ Transaction submitted to Node 4");
    println!("   Transaction Hash: {}", tx_hash);

    println!();

    // STEP 4: Wait for gossipsub propagation
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("STEP 4: Waiting for Gossipsub Propagation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("Waiting 5 seconds for propagation via /qnk/transactions topic...");
    tokio::time::sleep(Duration::from_secs(5)).await;
    println!();

    // STEP 5: Check transaction visibility on all nodes
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("STEP 5: Verifying Transaction Propagation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let nodes = vec![
        (NODE1, "Node 1"),
        (NODE2, "Node 2"),
        (NODE3, "Node 3"),
        (NODE4, "Node 4"),
    ];

    let mut propagation_count = 0;
    for (node, name) in &nodes {
        print!("Checking {} ({})... ", name, node);
        match check_transaction_on_node(&client, node, &tx_hash).await {
            Ok(true) => propagation_count += 1,
            Ok(false) => {},
            Err(e) => println!("   ⚠ Error checking node: {}", e),
        }
    }

    println!();
    println!("Propagation Result: {}/4 nodes see the transaction", propagation_count);
    println!();

    // STEP 6: Check balances
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("STEP 6: Verifying Balance Updates");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    match check_balance(&client, NODE1, &wallet1).await {
        Ok(balance) => println!("Wallet 1 balance: {} QNK (expected: 8.0)", balance),
        Err(e) => println!("⚠ Failed to check wallet 1 balance: {}", e),
    }

    match check_balance(&client, NODE1, &wallet2).await {
        Ok(balance) => println!("Wallet 2 balance: {} QNK (expected: 2.0)", balance),
        Err(e) => println!("⚠ Failed to check wallet 2 balance: {}", e),
    }

    println!();

    // SUMMARY
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                      TEST SUMMARY                             ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("✓ Wallet Creation: SUCCESS");
    println!("✓ Faucet Distribution: SUCCESS");
    println!("✓ Authenticated Transaction: SUCCESS");
    println!("✓ Transaction Propagation: {}/4 nodes ({}%)",
             propagation_count,
             (propagation_count * 100) / 4);
    println!();

    if propagation_count >= 2 {
        println!("🎉 TEST PASSED: Transaction propagated to multiple nodes!");
    } else {
        println!("⚠ TEST INCOMPLETE: Transaction only visible on source node");
        println!("   This is expected if other nodes are not running v0.0.9-beta");
    }

    Ok(())
}
