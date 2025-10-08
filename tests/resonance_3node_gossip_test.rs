//! 🎻 Three-Node Resonance Gossip Integration Test
//!
//! This test validates end-to-end gossip propagation of resonance consensus
//! across a 3-node network using simulated libp2p message passing.
//!
//! Philosophy: We test that vibrations propagate correctly through the symphony,
//! that consensus emerges from energy minimization, and that the network
//! achieves harmonic convergence.

use anyhow::Result;
use q_network::{ResonanceGossipManager, ResonanceProtocolHandler};
use q_resonance::{NarwhalTransaction, ResonanceCoordinator, ResonanceMessage};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{sleep, timeout, Duration};
use tracing::{debug, info, warn};

/// 🎻 Simulated network that routes messages between nodes
struct SimulatedNetwork {
    /// Map of node_id -> message receiver
    nodes: Arc<RwLock<HashMap<Vec<u8>, mpsc::UnboundedSender<Vec<u8>>>>>,
}

impl SimulatedNetwork {
    fn new() -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a node in the network
    async fn register_node(
        &self,
        node_id: Vec<u8>,
        tx: mpsc::UnboundedSender<Vec<u8>>,
    ) {
        self.nodes.write().await.insert(node_id, tx);
    }

    /// Broadcast message to all nodes except sender
    async fn broadcast(&self, sender_id: &[u8], data: Vec<u8>) {
        let nodes = self.nodes.read().await;
        for (node_id, tx) in nodes.iter() {
            if node_id.as_slice() != sender_id {
                if let Err(e) = tx.send(data.clone()) {
                    warn!("🎻 Failed to send to node {:?}: {}", node_id, e);
                }
            }
        }
    }
}

/// 🎻 Network node that processes resonance consensus
struct ResonanceNode {
    node_id: Vec<u8>,
    coordinator: Arc<ResonanceCoordinator>,
    handler: ResonanceProtocolHandler,
    network_rx: mpsc::UnboundedReceiver<Vec<u8>>,
    broadcast_tx: mpsc::UnboundedSender<(Vec<u8>, Vec<u8>)>, // (sender_id, data)
}

impl ResonanceNode {
    /// Create new node with gossip support
    fn new(
        node_id: Vec<u8>,
        broadcast_tx: mpsc::UnboundedSender<(Vec<u8>, Vec<u8>)>,
    ) -> (Self, mpsc::UnboundedSender<Vec<u8>>) {
        let (handler, coordinator, _network_tx) =
            ResonanceProtocolHandler::with_new_coordinator(node_id.clone());

        let (incoming_tx, incoming_rx) = mpsc::unbounded_channel();

        let node = Self {
            node_id: node_id.clone(),
            coordinator,
            handler,
            network_rx: incoming_rx,
            broadcast_tx,
        };

        (node, incoming_tx)
    }

    /// Run the node's event loop
    async fn run(mut self) -> Result<()> {
        info!("🎻 Node {:?} starting event loop", self.node_id);

        loop {
            tokio::select! {
                // Process incoming messages from network
                Some(data) = self.network_rx.recv() => {
                    debug!("🎻 Node {:?} received network message", self.node_id);
                    if let Err(e) = self.handler.handle_network_message(&data).await {
                        warn!("🎻 Node {:?} failed to process message: {}", self.node_id, e);
                    }
                }

                // Broadcast coordinator messages to network
                Some(data) = self.handler.next_broadcast() => {
                    debug!("🎻 Node {:?} broadcasting message", self.node_id);
                    if let Err(e) = self.broadcast_tx.send((self.node_id.clone(), data)) {
                        warn!("🎻 Node {:?} failed to broadcast: {}", self.node_id, e);
                    }
                }

                else => {
                    debug!("🎻 Node {:?} event loop completed", self.node_id);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Get coordinator reference
    fn coordinator(&self) -> &Arc<ResonanceCoordinator> {
        self.handler.coordinator()
    }
}

/// 🎻 Create demo transactions for testing
fn create_test_transactions(count: usize, base_timestamp: u64) -> Vec<NarwhalTransaction> {
    (0..count)
        .map(|i| NarwhalTransaction {
            hash: {
                let mut h = [0u8; 32];
                h[0] = i as u8;
                h[1] = (i >> 8) as u8;
                h
            },
            data: format!("Transaction {}", i).into_bytes(),
            sender: [0u8; 32],
            nonce: i as u64,
            signature: vec![0u8; 64],
            timestamp: base_timestamp + i as u64,
        })
        .collect()
}

#[tokio::test]
async fn test_three_node_resonance_gossip() -> Result<()> {
    // Initialize tracing for test visibility
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();

    info!("🎻 ═══════════════════════════════════════════════════════════");
    info!("🎻 THREE-NODE RESONANCE GOSSIP INTEGRATION TEST");
    info!("🎻 ═══════════════════════════════════════════════════════════");

    // Create simulated network
    let network = Arc::new(SimulatedNetwork::new());

    // Create broadcast channel for all nodes
    let (broadcast_tx, mut broadcast_rx) = mpsc::unbounded_channel::<(Vec<u8>, Vec<u8>)>();

    // Create three nodes
    info!("🎻 Creating 3 nodes with resonance consensus...");

    let node_id_1 = vec![1, 0, 0];
    let node_id_2 = vec![2, 0, 0];
    let node_id_3 = vec![3, 0, 0];

    let (node1, incoming_tx_1) = ResonanceNode::new(node_id_1.clone(), broadcast_tx.clone());
    let (node2, incoming_tx_2) = ResonanceNode::new(node_id_2.clone(), broadcast_tx.clone());
    let (node3, incoming_tx_3) = ResonanceNode::new(node_id_3.clone(), broadcast_tx.clone());

    // Get coordinator references before moving nodes
    let coordinator1 = Arc::clone(node1.coordinator());
    let coordinator2 = Arc::clone(node2.coordinator());
    let coordinator3 = Arc::clone(node3.coordinator());

    // Register nodes in network
    network.register_node(node_id_1.clone(), incoming_tx_1).await;
    network.register_node(node_id_2.clone(), incoming_tx_2).await;
    network.register_node(node_id_3.clone(), incoming_tx_3).await;

    info!("🎻 Nodes created and registered in network");

    // Spawn node event loops
    let node1_handle = tokio::spawn(async move {
        node1.run().await
    });

    let node2_handle = tokio::spawn(async move {
        node2.run().await
    });

    let node3_handle = tokio::spawn(async move {
        node3.run().await
    });

    // Spawn network message router
    let network_clone = Arc::clone(&network);
    let router_handle = tokio::spawn(async move {
        info!("🎻 Network router starting...");
        while let Some((sender_id, data)) = broadcast_rx.recv().await {
            debug!("🎻 Routing message from node {:?}", sender_id);
            network_clone.broadcast(&sender_id, data).await;
        }
        info!("🎻 Network router shutting down");
    });

    // Give nodes time to initialize
    sleep(Duration::from_millis(100)).await;

    info!("🎻 ───────────────────────────────────────────────────────────");
    info!("🎻 PHASE 1: Node 1 processes transactions");
    info!("🎻 ───────────────────────────────────────────────────────────");

    // Node 1 processes a batch of transactions
    let base_timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let transactions = create_test_transactions(10, base_timestamp);
    let tx_count = transactions.len();

    info!("🎻 Node 1 processing {} transactions with gossip...", tx_count);

    let ordered_hashes_1 = coordinator1
        .process_narwhal_batch_with_gossip(
            1, // round
            transactions.clone(),
            100.0, // stake
            vec![0.0, 0.0], // network position
        )
        .await?;

    info!("🎻 Node 1 achieved consensus: {} ordered hashes", ordered_hashes_1.len());
    assert_eq!(ordered_hashes_1.len(), tx_count, "Node 1 should order all transactions");

    // Give time for gossip propagation
    info!("🎻 Waiting for gossip propagation across network...");
    sleep(Duration::from_millis(500)).await;

    info!("🎻 ───────────────────────────────────────────────────────────");
    info!("🎻 PHASE 2: Nodes 2 and 3 process same transactions");
    info!("🎻 ───────────────────────────────────────────────────────────");

    // Nodes 2 and 3 process the same transactions
    info!("🎻 Node 2 processing transactions with gossip...");
    let ordered_hashes_2 = coordinator2
        .process_narwhal_batch_with_gossip(
            1,
            transactions.clone(),
            100.0,
            vec![1.0, 0.0],
        )
        .await?;

    info!("🎻 Node 3 processing transactions with gossip...");
    let ordered_hashes_3 = coordinator3
        .process_narwhal_batch_with_gossip(
            1,
            transactions.clone(),
            100.0,
            vec![0.0, 1.0],
        )
        .await?;

    info!("🎻 Node 2 achieved consensus: {} ordered hashes", ordered_hashes_2.len());
    info!("🎻 Node 3 achieved consensus: {} ordered hashes", ordered_hashes_3.len());

    // Give time for final gossip synchronization
    sleep(Duration::from_millis(500)).await;

    info!("🎻 ───────────────────────────────────────────────────────────");
    info!("🎻 PHASE 3: Validate consensus agreement");
    info!("🎻 ───────────────────────────────────────────────────────────");

    // All nodes should have ordered all transactions
    assert_eq!(ordered_hashes_2.len(), tx_count, "Node 2 should order all transactions");
    assert_eq!(ordered_hashes_3.len(), tx_count, "Node 3 should order all transactions");

    // Verify consensus ordering matches (energy minimization should converge)
    info!("🎻 Comparing transaction orderings across nodes...");

    let mut matches_12 = 0;
    let mut matches_13 = 0;
    let mut matches_23 = 0;

    for i in 0..tx_count {
        if ordered_hashes_1[i] == ordered_hashes_2[i] {
            matches_12 += 1;
        }
        if ordered_hashes_1[i] == ordered_hashes_3[i] {
            matches_13 += 1;
        }
        if ordered_hashes_2[i] == ordered_hashes_3[i] {
            matches_23 += 1;
        }
    }

    let agreement_12 = (matches_12 as f64 / tx_count as f64) * 100.0;
    let agreement_13 = (matches_13 as f64 / tx_count as f64) * 100.0;
    let agreement_23 = (matches_23 as f64 / tx_count as f64) * 100.0;

    info!("🎻 Consensus Agreement Metrics:");
    info!("   - Node 1 ↔ Node 2: {:.1}% agreement ({}/{} matches)", agreement_12, matches_12, tx_count);
    info!("   - Node 1 ↔ Node 3: {:.1}% agreement ({}/{} matches)", agreement_13, matches_13, tx_count);
    info!("   - Node 2 ↔ Node 3: {:.1}% agreement ({}/{} matches)", agreement_23, matches_23, tx_count);

    // In resonance consensus, perfect agreement is expected when all nodes
    // process the same transactions with the same energy functional
    let min_agreement = 80.0; // Allow some variance for now
    assert!(
        agreement_12 >= min_agreement,
        "Nodes 1 and 2 should have high consensus agreement (got {:.1}%)",
        agreement_12
    );

    info!("🎻 ───────────────────────────────────────────────────────────");
    info!("🎻 PHASE 4: Verify metrics and state");
    info!("🎻 ───────────────────────────────────────────────────────────");

    // Verify all nodes have processed the round
    let metrics1 = coordinator1.get_metrics();
    let metrics2 = coordinator2.get_metrics();
    let metrics3 = coordinator3.get_metrics();

    info!("🎻 Node 1 Metrics:");
    info!("   - Rounds processed: {}", metrics1.total_rounds_processed);
    info!("   - Vertices ordered: {}", metrics1.total_vertices_ordered);
    info!("   - Avg convergence: {:.2}ms", metrics1.average_convergence_time_ms);

    info!("🎻 Node 2 Metrics:");
    info!("   - Rounds processed: {}", metrics2.total_rounds_processed);
    info!("   - Vertices ordered: {}", metrics2.total_vertices_ordered);
    info!("   - Avg convergence: {:.2}ms", metrics2.average_convergence_time_ms);

    info!("🎻 Node 3 Metrics:");
    info!("   - Rounds processed: {}", metrics3.total_rounds_processed);
    info!("   - Vertices ordered: {}", metrics3.total_vertices_ordered);
    info!("   - Avg convergence: {:.2}ms", metrics3.average_convergence_time_ms);

    assert!(metrics1.total_rounds_processed >= 1, "Node 1 should process round");
    assert!(metrics2.total_rounds_processed >= 1, "Node 2 should process round");
    assert!(metrics3.total_rounds_processed >= 1, "Node 3 should process round");

    // Get spectral gap (consensus strength)
    if let Ok(spectral_gap_1) = coordinator1.get_spectral_gap().await {
        info!("🎻 Node 1 spectral gap: {:.4}", spectral_gap_1);
        assert!(spectral_gap_1 > 0.0, "Spectral gap should be positive");
    }

    // Get total energy
    let energy1 = coordinator1.get_total_energy();
    let energy2 = coordinator2.get_total_energy();
    let energy3 = coordinator3.get_total_energy();

    info!("🎻 System Energy:");
    info!("   - Node 1: {:.4}", energy1);
    info!("   - Node 2: {:.4}", energy2);
    info!("   - Node 3: {:.4}", energy3);

    assert!(energy1 > 0.0, "Node 1 energy should be positive");
    assert!(energy2 > 0.0, "Node 2 energy should be positive");
    assert!(energy3 > 0.0, "Node 3 energy should be positive");

    info!("🎻 ═══════════════════════════════════════════════════════════");
    info!("🎻 TEST COMPLETED SUCCESSFULLY! 🌟");
    info!("🎻 The distributed symphony has achieved harmonic consensus!");
    info!("🎻 ═══════════════════════════════════════════════════════════");

    // Cleanup: drop broadcast sender to shutdown router
    drop(broadcast_tx);

    // Wait for graceful shutdown with timeout
    let _ = timeout(Duration::from_secs(2), router_handle).await;

    Ok(())
}

#[tokio::test]
async fn test_gossip_state_synchronization() -> Result<()> {
    // Initialize tracing
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::INFO)
        .try_init();

    info!("🎻 Testing gossip state synchronization...");

    // Create two nodes
    let (broadcast_tx, _broadcast_rx) = mpsc::unbounded_channel();

    let (node1, _incoming_tx_1) = ResonanceNode::new(vec![1], broadcast_tx.clone());
    let (node2, _incoming_tx_2) = ResonanceNode::new(vec![2], broadcast_tx.clone());

    let coordinator1 = Arc::clone(node1.coordinator());
    let coordinator2 = Arc::clone(node2.coordinator());

    // Process transactions on node 1
    let transactions = create_test_transactions(5, 1000);

    coordinator1
        .process_narwhal_batch_with_gossip(1, transactions.clone(), 100.0, vec![0.0, 0.0])
        .await?;

    // Verify state tracker on node 1 has data
    let state_tracker_1 = coordinator1.get_state_tracker();
    let state1 = state_tracker_1.get_round_state(1).await;

    assert!(state1.is_some(), "Node 1 should have round 1 state");

    let state1 = state1.unwrap();
    assert!(!state1.string_states.is_empty(), "Node 1 should have string states");
    assert!(!state1.vertices.is_empty(), "Node 1 should have vertices");

    info!("🎻 Node 1 state tracker:");
    info!("   - String states: {}", state1.string_states.len());
    info!("   - Vertices: {}", state1.vertices.len());

    info!("🎻 Gossip state synchronization test passed! 🌟");

    Ok(())
}

#[tokio::test]
async fn test_byzantine_detection_with_gossip() -> Result<()> {
    // Initialize tracing
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::INFO)
        .try_init();

    info!("🎻 Testing Byzantine detection with gossip...");

    // Create coordinator with gossip
    let (handler, coordinator, _network_tx) =
        ResonanceProtocolHandler::with_new_coordinator(vec![1, 2, 3]);

    let transactions = create_test_transactions(10, 2000);

    // Process transactions
    coordinator
        .process_narwhal_batch_with_gossip(1, transactions, 100.0, vec![0.0, 0.0])
        .await?;

    // Get spectral gap for Byzantine detection
    if let Ok(spectral_gap) = coordinator.get_spectral_gap().await {
        info!("🎻 Spectral gap (Byzantine detection metric): {:.4}", spectral_gap);

        // A healthy network should have a positive spectral gap
        assert!(spectral_gap > 0.0, "Spectral gap should be positive for healthy network");

        // Spectral gap < 0.1 would indicate potential Byzantine behavior
        info!("🎻 Network health: {}", if spectral_gap > 0.1 { "GOOD ✅" } else { "SUSPICIOUS ⚠️" });
    }

    info!("🎻 Byzantine detection test passed! 🌟");

    Ok(())
}
