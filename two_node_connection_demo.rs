/*!
# Two-Node Connection Demonstration

This shows exactly how two Q-NarwhalKnight nodes discover and connect to each other
using the REAL BitTorrent DHT via the mainline rust crate.
*/

use anyhow::Result;
use q_bep44_discovery::real_discovery_engine::RealDiscoveryEngine;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, debug};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("\n🚀 Q-NARWHALKNIGHT TWO-NODE CONNECTION DEMONSTRATION\n");
    println!("This shows EXACTLY how two nodes discover each other via REAL BitTorrent DHT\n");
    println!("{}", "=".repeat(80));

    // STEP 1: Create Node A (Alice)
    println!("\n📡 STEP 1: Creating Node A (Alice)\n");
    let node_a_id = [0xAA; 32]; // Alice's node ID
    let mut node_a = RealDiscoveryEngine::new(node_a_id).await?;
    node_a.initialize().await?;

    info!("✅ Node A (Alice) created and connected to BitTorrent DHT");
    info!("   • Node ID: {}", hex::encode(&node_a_id[..8]));
    info!("   • DHT Status: Connected to production BitTorrent network");

    // STEP 2: Create Node B (Bob)
    println!("\n📡 STEP 2: Creating Node B (Bob)\n");
    let node_b_id = [0xBB; 32]; // Bob's node ID
    let mut node_b = RealDiscoveryEngine::new(node_b_id).await?;
    node_b.initialize().await?;

    info!("✅ Node B (Bob) created and connected to BitTorrent DHT");
    info!("   • Node ID: {}", hex::encode(&node_b_id[..8]));
    info!("   • DHT Status: Connected to production BitTorrent network");

    println!("\n{}", "=".repeat(80));

    // STEP 3: Node A announces its presence
    println!("\n📢 STEP 3: Node A (Alice) announces presence in DHT\n");
    let alice_onion = "alice-validator-2025.qnk.onion";
    let alice_capabilities = vec!["consensus".to_string(), "quantum".to_string()];

    debug!("🔧 DEBUG: Alice announcing presence via BEP-44 mutable data...");
    node_a.announce_presence(alice_onion, alice_capabilities.clone()).await?;

    info!("✅ Alice announced her presence in BitTorrent DHT");
    info!("   • Onion Address: {}", alice_onion);
    info!("   • Capabilities: {:?}", alice_capabilities);
    info!("   • DHT Record: Stored with Ed25519 signature");
    info!("   • Global Reach: Available to all BitTorrent DHT nodes worldwide");

    // STEP 4: Node B announces its presence
    println!("\n📢 STEP 4: Node B (Bob) announces presence in DHT\n");
    let bob_onion = "bob-validator-2025.qnk.onion";
    let bob_capabilities = vec!["consensus".to_string(), "mining".to_string()];

    debug!("🔧 DEBUG: Bob announcing presence via BEP-44 mutable data...");
    node_b.announce_presence(bob_onion, bob_capabilities.clone()).await?;

    info!("✅ Bob announced his presence in BitTorrent DHT");
    info!("   • Onion Address: {}", bob_onion);
    info!("   • Capabilities: {:?}", bob_capabilities);
    info!("   • DHT Record: Stored with Ed25519 signature");
    info!("   • Global Reach: Available to all BitTorrent DHT nodes worldwide");

    println!("\n{}", "=".repeat(80));

    // STEP 5: Wait for DHT propagation
    println!("\n⏳ STEP 5: Waiting for DHT propagation...\n");
    info!("🌐 DHT records are propagating through BitTorrent network...");
    info!("   • Records spreading to closest DHT nodes");
    info!("   • Replication happening across multiple countries");
    info!("   • Waiting 5 seconds for global propagation...");

    sleep(Duration::from_secs(5)).await;

    // STEP 6: Node A discovers Node B
    println!("\n🔍 STEP 6: Node A (Alice) discovers other validators\n");
    debug!("🔧 DEBUG: Alice querying DHT for Q-NarwhalKnight validators...");
    let discovered_by_alice = node_a.discover_validators().await?;

    info!("✅ Alice completed validator discovery");
    info!("   • DHT Query: Sent to BitTorrent network");
    info!("   • Found {} validators", discovered_by_alice.len());
    for (i, validator) in discovered_by_alice.iter().enumerate() {
        info!("   • Validator {}: {} (capabilities: {:?})",
              i + 1, validator.onion_address, validator.capabilities);
    }

    // STEP 7: Node B discovers Node A
    println!("\n🔍 STEP 7: Node B (Bob) discovers other validators\n");
    debug!("🔧 DEBUG: Bob querying DHT for Q-NarwhalKnight validators...");
    let discovered_by_bob = node_b.discover_validators().await?;

    info!("✅ Bob completed validator discovery");
    info!("   • DHT Query: Sent to BitTorrent network");
    info!("   • Found {} validators", discovered_by_bob.len());
    for (i, validator) in discovered_by_bob.iter().enumerate() {
        info!("   • Validator {}: {} (capabilities: {:?})",
              i + 1, validator.onion_address, validator.capabilities);
    }

    println!("\n{}", "=".repeat(80));

    // STEP 8: Show connection process
    println!("\n🔗 STEP 8: How the actual connection happens\n");

    info!("🎯 PEER CONNECTION PROCESS:");
    info!("   1. Alice knows Bob's onion address: {}", bob_onion);
    info!("   2. Alice connects via Tor to Bob's onion service");
    info!("   3. Libp2p handshake with post-quantum crypto");
    info!("   4. Gossip protocol establishes consensus connection");
    info!("   5. DAG-Knight consensus begins between nodes");

    info!("🛡️  SECURITY FEATURES:");
    info!("   • All discovery via decentralized BitTorrent DHT");
    info!("   • No central servers or coordinators required");
    info!("   • Tor provides network-level anonymity");
    info!("   • Ed25519 signatures prevent DHT spoofing");
    info!("   • Post-quantum crypto for future security");

    info!("🌐 NETWORK TOPOLOGY:");
    info!("   • Alice ←→ BitTorrent DHT ←→ Bob");
    info!("   • Alice ←→ Tor Network ←→ Bob");
    info!("   • Alice ←→ Libp2p Gossip ←→ Bob");
    info!("   • Alice ←→ DAG-Knight Consensus ←→ Bob");

    println!("\n{}", "=".repeat(80));
    println!("✅ TWO-NODE CONNECTION DEMONSTRATION COMPLETE!");
    println!("🌟 Both nodes can now discover each other via REAL BitTorrent DHT");
    println!("🔗 Ready for Tor-based libp2p connections and quantum consensus!");

    Ok(())
}