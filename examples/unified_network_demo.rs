/// Unified Network Manager Demo - Sophisticated Multi-Layer Integration
/// 
/// This example demonstrates how the Q-NarwhalKnight networking layers work together
/// intelligently to provide optimal routing, automatic failover, and adaptive behavior
/// based on network conditions and message requirements.

use anyhow::Result;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn, debug};
use uuid::Uuid;

use q_network::unified_network_manager::{
    UnifiedNetworkManager, UnifiedNetworkManagerBuilder, NetworkLayer, 
    MessageClass, RoutingStrategy, NetworkCommand, UnifiedNetworkEvent,
};
use q_types::NodeId;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("debug,libp2p=warn,hyper=warn")
        .init();

    info!("🚀 Starting Q-NarwhalKnight Unified Network Integration Demo");
    
    // Create multiple nodes to demonstrate network interactions
    let node_configs = vec![
        ([0x01; 32], "Alpha Node"),
        ([0x02; 32], "Beta Node"), 
        ([0x03; 32], "Gamma Node"),
    ];
    
    let mut network_managers = Vec::new();
    let mut command_senders = Vec::new();
    let mut event_receivers = Vec::new();
    
    // Initialize network managers for each node
    for (node_id, name) in node_configs {
        info!("🔧 Initializing {} with ID: {}", name, hex::encode(&node_id[..4]));
        
        let mut manager = UnifiedNetworkManagerBuilder::new(node_id)
            .routing_strategy(RoutingStrategy::Adaptive)
            .enable_layer(NetworkLayer::LibP2P, true)
            .enable_layer(NetworkLayer::Tor, true)
            .enable_layer(NetworkLayer::DNSPhantom, true)
            .enable_layer(NetworkLayer::BitTorrentDHT, true)
            .health_check_interval(Duration::from_secs(10))
            .build()
            .await?;
        
        // Initialize the network layers
        manager.initialize().await?;
        
        // Get communication handles
        let command_sender = manager.command_sender();
        let event_receiver = manager.subscribe_events();
        
        command_senders.push(command_sender);
        event_receivers.push(event_receiver);
        
        // Start the manager in the background
        let name_clone = name.to_string();
        tokio::spawn(async move {
            if let Err(e) = manager.start().await {
                warn!("❌ {} network manager failed: {}", name_clone, e);
            }
        });
        
        // Brief delay between node startups
        sleep(Duration::from_secs(1)).await;
    }
    
    info!("✅ All network managers initialized and started");
    
    // Demonstrate intelligent routing scenarios
    demonstrate_routing_scenarios(&command_senders).await?;
    
    // Monitor network events
    monitor_network_events(event_receivers).await?;
    
    Ok(())
}

/// Demonstrate various routing scenarios showcasing intelligent layer selection
async fn demonstrate_routing_scenarios(
    command_senders: &[tokio::sync::mpsc::Sender<NetworkCommand>]
) -> Result<()> {
    info!("🎯 Demonstrating Intelligent Routing Scenarios");
    
    let alpha_sender = &command_senders[0];
    let beta_node_id = [0x02; 32];
    
    // Scenario 1: Urgent Consensus Message (should use fastest transport)
    info!("\n📈 Scenario 1: Urgent Consensus Message");
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        alpha_sender.send(NetworkCommand::SendMessage {
            target: Some(beta_node_id),
            content: b"URGENT: New block proposal #12345".to_vec(),
            message_class: MessageClass::UrgentConsensus,
            response_channel: tx,
        }).await?;
        
        match rx.await? {
            Ok(message_id) => info!("✅ Urgent message sent: {}", message_id),
            Err(e) => warn!("❌ Failed to send urgent message: {}", e),
        }
    }
    
    sleep(Duration::from_millis(500)).await;
    
    // Scenario 2: Private Message (should prefer Tor/DNS Phantom)
    info!("\n🔒 Scenario 2: Private Message");
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        alpha_sender.send(NetworkCommand::SendMessage {
            target: Some(beta_node_id),
            content: b"PRIVATE: Sensitive validator communication".to_vec(),
            message_class: MessageClass::PrivateMessage,
            response_channel: tx,
        }).await?;
        
        match rx.await? {
            Ok(message_id) => info!("✅ Private message sent: {}", message_id),
            Err(e) => warn!("❌ Failed to send private message: {}", e),
        }
    }
    
    sleep(Duration::from_millis(500)).await;
    
    // Scenario 3: Block Propagation (balanced approach)
    info!("\n📦 Scenario 3: Block Propagation");
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        alpha_sender.send(NetworkCommand::SendMessage {
            target: None, // Broadcast to all peers
            content: b"BLOCK: New confirmed block with 1000 transactions".to_vec(),
            message_class: MessageClass::BlockPropagation,
            response_channel: tx,
        }).await?;
        
        match rx.await? {
            Ok(message_id) => info!("✅ Block broadcast sent: {}", message_id),
            Err(e) => warn!("❌ Failed to broadcast block: {}", e),
        }
    }
    
    sleep(Duration::from_millis(500)).await;
    
    // Scenario 4: Emergency Broadcast (use all available channels)
    info!("\n🚨 Scenario 4: Emergency Broadcast");
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        alpha_sender.send(NetworkCommand::SendMessage {
            target: None,
            content: b"EMERGENCY: Network partition detected, initiating recovery".to_vec(),
            message_class: MessageClass::Emergency,
            response_channel: tx,
        }).await?;
        
        match rx.await? {
            Ok(message_id) => info!("✅ Emergency broadcast sent: {}", message_id),
            Err(e) => warn!("❌ Failed to send emergency broadcast: {}", e),
        }
    }
    
    sleep(Duration::from_millis(500)).await;
    
    // Scenario 5: Force specific layer usage
    info!("\n🎚️ Scenario 5: Force DNS Phantom Usage");
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        alpha_sender.send(NetworkCommand::SendViaLayer {
            layer: NetworkLayer::DNSPhantom,
            target: Some(beta_node_id),
            content: b"STEGANOGRAPHIC: Hidden message via DNS queries".to_vec(),
            response_channel: tx,
        }).await?;
        
        match rx.await? {
            Ok(message_id) => info!("✅ DNS Phantom message sent: {}", message_id),
            Err(e) => warn!("❌ Failed to send via DNS Phantom: {}", e),
        }
    }
    
    sleep(Duration::from_millis(500)).await;
    
    // Scenario 6: Change routing strategy dynamically
    info!("\n🔄 Scenario 6: Dynamic Strategy Change");
    {
        alpha_sender.send(NetworkCommand::SetRoutingStrategy {
            strategy: RoutingStrategy::MaxPrivacy,
        }).await?;
        
        info!("🔒 Switched to maximum privacy routing");
        
        let (tx, rx) = tokio::sync::oneshot::channel();
        alpha_sender.send(NetworkCommand::SendMessage {
            target: Some(beta_node_id),
            content: b"TEST: Message after privacy strategy change".to_vec(),
            message_class: MessageClass::BlockPropagation,
            response_channel: tx,
        }).await?;
        
        match rx.await? {
            Ok(message_id) => info!("✅ Privacy-mode message sent: {}", message_id),
            Err(e) => warn!("❌ Failed to send privacy message: {}", e),
        }
    }
    
    sleep(Duration::from_millis(500)).await;
    
    // Scenario 7: Get network health metrics
    info!("\n📊 Scenario 7: Network Health Analysis");
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        alpha_sender.send(NetworkCommand::GetHealthMetrics {
            response_channel: tx,
        }).await?;
        
        let metrics = rx.await?;
        info!("📈 Current Network Health:");
        for (layer, metric) in metrics {
            info!("  {:?}: Available={}, Latency={:?}ms, Success Rate={:.2}%",
                  layer, metric.is_available, metric.latency_ms, metric.success_rate * 100.0);
        }
    }
    
    sleep(Duration::from_millis(500)).await;
    
    // Scenario 8: Get discovered peers
    info!("\n👥 Scenario 8: Peer Discovery Status");
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        alpha_sender.send(NetworkCommand::GetDiscoveredPeers {
            response_channel: tx,
        }).await?;
        
        let peers = rx.await?;
        info!("🔍 Discovered Peers: {} total", peers.len());
        for (node_id, connections) in peers {
            info!("  Node {}: {} connection methods", 
                  hex::encode(&node_id[..4]), connections.len());
            for conn in connections {
                info!("    {:?}: {} (latency: {:?}ms, success: {:.1}%)",
                      conn.layer, conn.address, conn.latency_ms, conn.success_rate * 100.0);
            }
        }
    }
    
    info!("✅ All routing scenarios completed successfully");
    Ok(())
}

/// Monitor network events to show real-time coordination
async fn monitor_network_events(
    mut event_receivers: Vec<tokio::sync::broadcast::Receiver<UnifiedNetworkEvent>>
) -> Result<()> {
    info!("👁️ Starting Network Event Monitoring");
    
    // Monitor events for 30 seconds
    let monitor_duration = Duration::from_secs(30);
    let start_time = tokio::time::Instant::now();
    
    let mut event_count = 0;
    
    while start_time.elapsed() < monitor_duration {
        // Try to receive events from any node
        for (node_idx, receiver) in event_receivers.iter_mut().enumerate() {
            match receiver.try_recv() {
                Ok(event) => {
                    event_count += 1;
                    let node_name = match node_idx {
                        0 => "Alpha",
                        1 => "Beta", 
                        2 => "Gamma",
                        _ => "Unknown",
                    };
                    
                    match event {
                        UnifiedNetworkEvent::PeerDiscovered { peer_id, layer, .. } => {
                            info!("🔍 {} discovered peer {} via {:?}", 
                                  node_name, hex::encode(&peer_id[..4]), layer);
                        }
                        
                        UnifiedNetworkEvent::MessageSent { message_id, layer, latency_ms, .. } => {
                            info!("📤 {} sent message {} via {:?} ({}ms)", 
                                  node_name, message_id, layer, latency_ms);
                        }
                        
                        UnifiedNetworkEvent::MessageReceived { message_id, from, layer, .. } => {
                            info!("📥 {} received message {} from {} via {:?}", 
                                  node_name, message_id, hex::encode(&from[..4]), layer);
                        }
                        
                        UnifiedNetworkEvent::RoutingDecision { message_class, selected_layers, reason } => {
                            info!("🧠 {} routing decision for {:?}: {:?} - {}", 
                                  node_name, message_class, selected_layers, reason);
                        }
                        
                        UnifiedNetworkEvent::FailoverTriggered { failed_layer, backup_layer, message_id } => {
                            warn!("🔄 {} failover: {:?} → {:?} for message {}", 
                                  node_name, failed_layer, backup_layer, message_id);
                        }
                        
                        UnifiedNetworkEvent::LayerHealthChanged { layer, metrics } => {
                            debug!("🏥 {} health update for {:?}: available={}, latency={:?}ms", 
                                   node_name, layer, metrics.is_available, metrics.latency_ms);
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                    // No events available, continue
                }
                Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => {
                    warn!("Event buffer overflow for node {}", node_idx);
                }
                Err(tokio::sync::broadcast::error::TryRecvError::Closed) => {
                    warn!("Event channel closed for node {}", node_idx);
                }
            }
        }
        
        // Brief delay to prevent busy waiting
        sleep(Duration::from_millis(100)).await;
    }
    
    info!("📊 Monitoring completed: {} events processed", event_count);
    Ok(())
}

/// Simulate network conditions and failures to test adaptive behavior
#[allow(dead_code)]
async fn simulate_network_conditions(
    command_senders: &[tokio::sync::mpsc::Sender<NetworkCommand>]
) -> Result<()> {
    info!("🌐 Simulating Network Conditions and Failures");
    
    let alpha_sender = &command_senders[0];
    
    // Simulate Tor network slowdown
    info!("🐌 Simulating Tor network slowdown...");
    // In a real implementation, this would actually affect the Tor layer
    sleep(Duration::from_secs(2)).await;
    
    // Test message sending during degraded conditions
    let (tx, rx) = tokio::sync::oneshot::channel();
    alpha_sender.send(NetworkCommand::SendMessage {
        target: Some([0x02; 32]),
        content: b"TEST: Message during Tor slowdown".to_vec(),
        message_class: MessageClass::UrgentConsensus,
        response_channel: tx,
    }).await?;
    
    match rx.await? {
        Ok(message_id) => info!("✅ Message sent despite Tor issues: {}", message_id),
        Err(e) => warn!("❌ Failed during Tor slowdown: {}", e),
    }
    
    // Simulate DNS censorship
    info!("🚫 Simulating DNS censorship...");
    // This would disable DNS Phantom layer in real implementation
    sleep(Duration::from_secs(2)).await;
    
    // Test fallback behavior
    let (tx, rx) = tokio::sync::oneshot::channel();
    alpha_sender.send(NetworkCommand::SendMessage {
        target: Some([0x02; 32]),
        content: b"TEST: Message during DNS censorship".to_vec(),
        message_class: MessageClass::PrivateMessage,
        response_channel: tx,
    }).await?;
    
    match rx.await? {
        Ok(message_id) => info!("✅ Message sent with fallback routing: {}", message_id),
        Err(e) => warn!("❌ Failed during DNS censorship: {}", e),
    }
    
    info!("✅ Network condition simulation completed");
    Ok(())
}

/// Demonstrate load balancing across multiple transport layers
#[allow(dead_code)]
async fn demonstrate_load_balancing(
    command_senders: &[tokio::sync::mpsc::Sender<NetworkCommand>]
) -> Result<()> {
    info!("⚖️ Demonstrating Load Balancing");
    
    let alpha_sender = &command_senders[0];
    
    // Switch to load balanced routing
    alpha_sender.send(NetworkCommand::SetRoutingStrategy {
        strategy: RoutingStrategy::LoadBalanced,
    }).await?;
    
    // Send multiple messages to see load distribution
    for i in 0..10 {
        let (tx, rx) = tokio::sync::oneshot::channel();
        alpha_sender.send(NetworkCommand::SendMessage {
            target: Some([0x02; 32]),
            content: format!("LOAD_TEST: Message #{}", i).into_bytes(),
            message_class: MessageClass::BlockPropagation,
            response_channel: tx,
        }).await?;
        
        match rx.await? {
            Ok(message_id) => debug!("📤 Load balanced message #{}: {}", i, message_id),
            Err(e) => warn!("❌ Load balanced message #{} failed: {}", i, e),
        }
        
        // Brief delay between messages
        sleep(Duration::from_millis(200)).await;
    }
    
    info!("✅ Load balancing demonstration completed");
    Ok(())
}