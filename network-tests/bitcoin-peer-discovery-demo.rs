use q_network::peer_discovery::PeerDiscovery;
use q_types::Phase;
use tokio;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    println!("🔍 Q-NarwhalKnight Bitcoin-Based Peer Discovery Demo");
    println!("=================================================");
    
    // Create peer discovery instance
    let mut peer_discovery = PeerDiscovery::new();
    
    // Set current phase to Phase 1 (post-quantum)
    peer_discovery.update_phase_capabilities(Phase::Phase1);
    
    println!("\n📋 Our Capabilities:");
    for cap in peer_discovery.get_capabilities() {
        println!("  - {}", cap);
    }
    
    println!("\n🔗 Starting Bitcoin-based peer discovery...");
    
    // Attempt Bitcoin bootstrap
    match peer_discovery.bootstrap_from_bitcoin().await {
        Ok(discovered_peers) => {
            println!("✅ Bitcoin bootstrap completed!");
            println!("📊 Discovered {} Q-NarwhalKnight peers:", discovered_peers.len());
            
            for peer in discovered_peers {
                println!("  🎯 Node: {} at {}", peer.node_id, peer.multiaddr);
                println!("     Capabilities: {:?}", peer.capabilities);
                if let Some(region) = &peer.region {
                    println!("     Region: {}", region);
                }
            }
            
            // Get all known peers
            let all_peers = peer_discovery.get_all_peers().await;
            println!("\n📈 Total known peers: {}", all_peers.len());
            
            // Filter by capabilities
            let quantum_peers = peer_discovery.get_quantum_peers().await;
            println!("🔬 Quantum-capable peers: {}", quantum_peers.len());
            
            let phase1_peers = peer_discovery.get_peers_by_phase(Phase::Phase1).await;
            println!("🚀 Phase 1 peers: {}", phase1_peers.len());
            
            println!("\n🎯 BITCOIN PEER DISCOVERY: SUCCESS");
            println!("✅ Successfully discovered Q-NarwhalKnight nodes through Bitcoin network");
            println!("✅ Peer registry populated with discovery results");
            println!("✅ Ready for quantum consensus participation");
        }
        Err(e) => {
            warn!("❌ Bitcoin bootstrap failed: {}", e);
            println!("⚠️ BITCOIN PEER DISCOVERY: FAILED");
            println!("🔧 Ensure Bitcoin node is running and accessible");
            println!("🔧 Check that Q-NarwhalKnight test nodes are active");
        }
    }
    
    println!("\n🏁 Demo completed");
    Ok(())
}