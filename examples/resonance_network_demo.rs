//! 🎻 Resonance Network Demo
//!
//! This example demonstrates how to integrate Quillon Resonance Consensus
//! with libp2p gossipsub for distributed harmonic consensus.
//!
//! Philosophy: This is the complete symphony in action - resonance consensus
//! propagating across the network like sound waves through air.

use anyhow::Result;
use q_network::{resonance_topic, ResonanceGossipManager, ResonanceProtocolHandler};
use q_resonance::{NarwhalTransaction, ResonanceCoordinator};
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    info!("🎻 Starting Resonance Network Demo");

    // Create node ID
    let node_id = vec![1, 2, 3];

    // Create resonance protocol handler with coordinator
    let (handler, coordinator, network_tx) =
        ResonanceProtocolHandler::with_new_coordinator(node_id.clone());

    info!("🎻 Resonance coordinator created for node {:?}", node_id);

    // Create gossip manager
    let mut manager = ResonanceGossipManager::new(handler);

    info!(
        "🎻 Gossip manager initialized for topic: {}",
        manager.topic()
    );

    // Simulate processing a batch of transactions
    let transactions = create_demo_transactions(5);

    info!("🎻 Processing {} transactions with resonance consensus", transactions.len());

    let ordered_hashes = coordinator
        .process_narwhal_batch_with_gossip(
            1, // round
            transactions,
            100.0, // stake
            vec![0.0, 0.0], // network position
        )
        .await?;

    info!(
        "🎻 Consensus achieved! Ordered {} transaction hashes",
        ordered_hashes.len()
    );

    // Get consensus metrics
    let metrics = coordinator.get_metrics();
    info!("🎻 Consensus Metrics:");
    info!("  - Total rounds processed: {}", metrics.total_rounds_processed);
    info!(
        "  - Average convergence time: {:.2}ms",
        metrics.average_convergence_time_ms
    );
    info!("  - Total vertices ordered: {}", metrics.total_vertices_ordered);

    // Get spectral gap (consensus strength)
    if let Ok(spectral_gap) = coordinator.get_spectral_gap().await {
        info!("  - Spectral gap (consensus strength): {:.4}", spectral_gap);
    }

    // Get total energy
    let total_energy = coordinator.get_total_energy();
    info!("  - Total energy: {:.4}", total_energy);

    // Demonstrate gossip broadcasting
    info!("🎻 Demonstrating gossip broadcast capabilities...");

    // In a real implementation, this would be wired to libp2p gossipsub
    // For this demo, we'll show the pattern
    info!(
        "🎻 Subscribe to topic in gossipsub: {}",
        manager.topic()
    );
    info!("🎻 Process incoming messages with: manager.handle_gossip_message()");
    info!("🎻 Broadcast coordinator messages with: manager.next_broadcast()");

    // Simulate waiting for network activity
    sleep(Duration::from_secs(1)).await;

    info!("🎻 Resonance Network Demo Complete!");
    info!("🎻 The distributed symphony has played successfully! 🌌");

    Ok(())
}

/// Create demo transactions for testing
fn create_demo_transactions(count: usize) -> Vec<NarwhalTransaction> {
    (0..count)
        .map(|i| NarwhalTransaction {
            hash: [i as u8; 32],
            data: format!("Transaction {}", i).into_bytes(),
            sender: [0u8; 32],
            nonce: i as u64,
            signature: vec![0u8; 64],
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64
                + i as u64,
        })
        .collect()
}
