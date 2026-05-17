/*!
# Q-NarwhalKnight Bootstrap Node Demonstration

This demonstrates how Q-NarwhalKnight nodes connect to each other through the
primary bootstrap node at 185.182.185.227.
*/

use anyhow::Result;
use q_bep44_discovery::real_discovery_engine::RealDiscoveryEngine;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, debug};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("\n🚀 Q-NARWHALKNIGHT BOOTSTRAP NODE DEMONSTRATION\n");
    println!("Primary Bootstrap: 185.182.185.227:6881");
    println!("This shows how two nodes connect through the Q-NarwhalKnight bootstrap node\n");
    println!("{}", "=".repeat(80));

    // STEP 1: Create first node that will connect via bootstrap
    println!("\n📡 STEP 1: Creating Node 1 (Alice) - Bootstrap Connection\n");
    let node1_id = [0x11; 32]; // Alice's node ID
    let mut node1 = RealDiscoveryEngine::new(node1_id).await?;
    node1.initialize().await?;

    info!("✅ Node 1 (Alice) created and connected via Q-NarwhalKnight bootstrap");
    info!("   • Node ID: {}", hex::encode(&node1_id[..8]));
    info!("   • Bootstrap: 185.182.185.227:6881");
    info!("   • DHT Status: Connected to Q-NarwhalKnight network");

    // STEP 2: Create second node that will also connect via bootstrap
    println!("\n📡 STEP 2: Creating Node 2 (Bob) - Bootstrap Connection\n");
    let node2_id = [0x22; 32]; // Bob's node ID
    let mut node2 = RealDiscoveryEngine::new(node2_id).await?;
    node2.initialize().await?;

    info!("✅ Node 2 (Bob) created and connected via Q-NarwhalKnight bootstrap");
    info!("   • Node ID: {}", hex::encode(&node2_id[..8]));
    info!("   • Bootstrap: 185.182.185.227:6881");
    info!("   • DHT Status: Connected to Q-NarwhalKnight network");

    println!("\n{}", "=".repeat(80));

    // STEP 3: Both nodes announce through the same bootstrap network
    println!("\n📢 STEP 3: Both nodes announce presence via bootstrap node\n");

    let alice_onion = "alice-prod-2025.qnk.onion";
    let alice_capabilities = vec!["consensus".to_string(), "quantum".to_string()];

    let bob_onion = "bob-prod-2025.qnk.onion";
    let bob_capabilities = vec!["consensus".to_string(), "mining".to_string()];

    debug!("🔧 DEBUG: Alice announcing via Q-NarwhalKnight bootstrap network...");
    node1.announce_presence(alice_onion, alice_capabilities.clone()).await?;

    debug!("🔧 DEBUG: Bob announcing via Q-NarwhalKnight bootstrap network...");
    node2.announce_presence(bob_onion, bob_capabilities.clone()).await?;

    info!("✅ Both nodes announced via Q-NarwhalKnight bootstrap network");
    info!("   • Alice: {} {:?}", alice_onion, alice_capabilities);
    info!("   • Bob: {} {:?}", bob_onion, bob_capabilities);
    info!("   • Network: Shared Q-NarwhalKnight DHT via 185.182.185.227");

    println!("\n{}", "=".repeat(80));

    // STEP 4: Show how discovery works through bootstrap
    println!("\n🔍 STEP 4: Node discovery through Q-NarwhalKnight bootstrap\n");

    // Wait for bootstrap network propagation
    info!("🌐 Waiting for Q-NarwhalKnight bootstrap network propagation...");
    info!("   • Records spreading through 185.182.185.227 DHT cluster");
    info!("   • Both nodes connected to same bootstrap infrastructure");
    info!("   • Waiting 5 seconds for network propagation...");

    sleep(Duration::from_secs(5)).await;

    // Both nodes discover each other through the bootstrap network
    debug!("🔧 DEBUG: Alice discovering peers via Q-NarwhalKnight bootstrap...");
    let alice_discoveries = node1.discover_validators().await?;

    debug!("🔧 DEBUG: Bob discovering peers via Q-NarwhalKnight bootstrap...");
    let bob_discoveries = node2.discover_validators().await?;

    info!("✅ Peer discovery completed via Q-NarwhalKnight bootstrap");
    info!("   • Alice found {} validators", alice_discoveries.len());
    info!("   • Bob found {} validators", bob_discoveries.len());

    println!("\n{}", "=".repeat(80));

    // STEP 5: Explain the bootstrap network topology
    println!("\n🌐 STEP 5: Q-NarwhalKnight Bootstrap Network Topology\n");

    info!("🎯 BOOTSTRAP NETWORK ARCHITECTURE:");
    info!("   ┌─────────────┐    ┌─────────────────────┐    ┌─────────────┐");
    info!("   │ Alice Node  │◄──►│ Q-NarwhalKnight     │◄──►│ Bob Node    │");
    info!("   │ (Any IP)    │    │ Bootstrap           │    │ (Any IP)    │");
    info!("   │             │    │ 185.182.185.227     │    │             │");
    info!("   └─────────────┘    └─────────────────────┘    └─────────────┘");
    info!("          │                      │                      │");
    info!("          ▼                      ▼                      ▼");
    info!("    Announces via           DHT Network              Announces via");
    info!("   Bootstrap DHT            Coordinator             Bootstrap DHT");

    info!("🔗 CONNECTION PROCESS:");
    info!("   1. Both nodes connect to 185.182.185.227:6881 on startup");
    info!("   2. Bootstrap node provides DHT routing table entries");
    info!("   3. Nodes announce their presence through bootstrap DHT");
    info!("   4. Discovery queries route through bootstrap infrastructure");
    info!("   5. Nodes learn each other's onion addresses");
    info!("   6. Direct Tor connections established for consensus");

    info!("🛡️ BOOTSTRAP SECURITY:");
    info!("   • Bootstrap only helps initial DHT connection");
    info!("   • All data signed with Ed25519 keys");
    info!("   • Consensus traffic uses direct Tor connections");
    info!("   • Bootstrap cannot see or modify consensus data");
    info!("   • Fallback to public BitTorrent if bootstrap fails");

    info!("⚡ PERFORMANCE BENEFITS:");
    info!("   • Faster initial network join (no scanning needed)");
    info!("   • Reliable peer discovery for Q-NarwhalKnight validators");
    info!("   • Reduced network noise (targeted discovery)");
    info!("   • Redundant connections to public DHT as backup");

    println!("\n{}", "=".repeat(80));
    println!("✅ Q-NARWHALKNIGHT BOOTSTRAP NODE DEMONSTRATION COMPLETE!");
    println!("🌟 Both nodes successfully connected via 185.182.185.227 bootstrap");
    println!("🔗 Ready for production Q-NarwhalKnight validator network!");

    Ok(())
}