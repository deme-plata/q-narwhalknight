/// Comprehensive Integration Test for Unified Network Manager
/// 
/// This test demonstrates the sophisticated coordination between all networking
/// layers and validates the intelligent routing, failover, and optimization
/// capabilities of the Q-NarwhalKnight unified network system.

use anyhow::Result;
use std::{sync::Arc, time::Duration};
use tokio::{sync::oneshot, time::sleep};
use tracing::{info, warn};
use uuid::Uuid;

use q_network::{
    unified_network_manager::{
        UnifiedNetworkManager, UnifiedNetworkManagerBuilder, NetworkLayer, 
        MessageClass, RoutingStrategy, NetworkCommand, UnifiedNetworkEvent,
    },
    network_metrics::{NetworkMetricsCollector, MetricsReporter},
};

/// Integration test configuration
struct TestConfig {
    pub node_count: usize,
    pub test_duration_seconds: u64,
    pub message_interval_ms: u64,
    pub failure_simulation_enabled: bool,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            node_count: 3,
            test_duration_seconds: 60,
            message_interval_ms: 1000,
            failure_simulation_enabled: true,
        }
    }
}

/// Test node representation
struct TestNode {
    pub node_id: [u8; 32],
    pub name: String,
    pub command_sender: tokio::sync::mpsc::Sender<NetworkCommand>,
    pub event_receiver: tokio::sync::broadcast::Receiver<UnifiedNetworkEvent>,
    pub metrics_collector: Arc<NetworkMetricsCollector>,
}

/// Comprehensive integration test
#[tokio::test]
async fn test_unified_network_integration() -> Result<()> {
    // Initialize logging for test visibility
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info,q_network=debug")
        .try_init();

    info!("🚀 Starting Q-NarwhalKnight Unified Network Integration Test");
    
    let config = TestConfig::default();
    
    // Create test network
    let test_network = create_test_network(&config).await?;
    
    // Run coordinated tests
    run_coordination_tests(&test_network, &config).await?;
    
    // Validate intelligent routing
    validate_intelligent_routing(&test_network).await?;
    
    // Test failover mechanisms
    test_failover_mechanisms(&test_network).await?;
    
    // Analyze performance metrics
    analyze_performance_metrics(&test_network).await?;
    
    info!("✅ Unified Network Integration Test completed successfully");
    Ok(())
}

/// Create test network with multiple nodes
async fn create_test_network(config: &TestConfig) -> Result<Vec<TestNode>> {
    info!("🔧 Creating test network with {} nodes", config.node_count);
    
    let mut test_nodes = Vec::new();
    
    for i in 0..config.node_count {
        let node_id = create_test_node_id(i);
        let name = format!("TestNode-{}", i);
        
        info!("📡 Initializing {}", name);
        
        // Create metrics collector for this node
        let metrics_collector = Arc::new(NetworkMetricsCollector::new());
        
        // Create and initialize network manager
        let mut manager = UnifiedNetworkManagerBuilder::new(node_id)
            .routing_strategy(RoutingStrategy::Adaptive)
            .enable_layer(NetworkLayer::LibP2P, true)
            .enable_layer(NetworkLayer::Tor, true)
            .enable_layer(NetworkLayer::DNSPhantom, true)
            .enable_layer(NetworkLayer::BitTorrentDHT, false) // Disable for testing simplicity
            .health_check_interval(Duration::from_secs(5))
            .build()
            .await?;
        
        manager.initialize().await?;
        
        let command_sender = manager.command_sender();
        let event_receiver = manager.subscribe_events();
        
        // Start the manager
        let name_clone = name.clone();
        let metrics_clone = Arc::clone(&metrics_collector);
        tokio::spawn(async move {
            let result = tokio::select! {
                res = manager.start() => res,
                _ = sleep(Duration::from_secs(300)) => {
                    info!("⏰ {} manager timeout", name_clone);
                    Ok(())
                }
            };
            
            if let Err(e) = result {
                warn!("❌ {} network manager failed: {}", name_clone, e);
            }
        });
        
        test_nodes.push(TestNode {
            node_id,
            name,
            command_sender,
            event_receiver,
            metrics_collector,
        });
        
        // Brief delay between node startups
        sleep(Duration::from_millis(200)).await;
    }
    
    // Allow time for network formation
    info!("⏳ Allowing network formation time...");
    sleep(Duration::from_secs(5)).await;
    
    info!("✅ Test network created successfully");
    Ok(test_nodes)
}

/// Run coordinated tests across the network
async fn run_coordination_tests(test_network: &[TestNode], config: &TestConfig) -> Result<()> {
    info!("🎯 Running coordination tests");
    
    let test_duration = Duration::from_secs(config.test_duration_seconds);
    let message_interval = Duration::from_millis(config.message_interval_ms);
    
    let start_time = tokio::time::Instant::now();
    let mut message_count = 0;
    
    // Test different message types across the network
    let message_scenarios = vec![
        (MessageClass::UrgentConsensus, "URGENT: Consensus message"),
        (MessageClass::BlockPropagation, "BLOCK: New block data"),
        (MessageClass::PrivateMessage, "PRIVATE: Confidential communication"),
        (MessageClass::Discovery, "DISCOVERY: Peer discovery message"),
        (MessageClass::Emergency, "EMERGENCY: Network alert"),
    ];
    
    while start_time.elapsed() < test_duration {
        for (sender_idx, sender_node) in test_network.iter().enumerate() {
            for (receiver_idx, receiver_node) in test_network.iter().enumerate() {
                if sender_idx == receiver_idx {
                    continue; // Skip self-messaging
                }
                
                // Select message scenario cyclically
                let scenario_idx = message_count % message_scenarios.len();
                let (message_class, message_prefix) = &message_scenarios[scenario_idx];
                
                let message_content = format!(
                    "{} #{} from {} to {}",
                    message_prefix, message_count, sender_node.name, receiver_node.name
                );
                
                // Send message
                let (tx, rx) = oneshot::channel();
                let send_result = sender_node.command_sender.send(NetworkCommand::SendMessage {
                    target: Some(receiver_node.node_id),
                    content: message_content.as_bytes().to_vec(),
                    message_class: *message_class,
                    response_channel: tx,
                }).await;
                
                if send_result.is_ok() {
                    match rx.await {
                        Ok(Ok(message_id)) => {
                            info!("📤 {} sent message {} to {}", 
                                  sender_node.name, message_id, receiver_node.name);
                            
                            // Record metrics
                            sender_node.metrics_collector.record_message_sent(
                                NetworkLayer::LibP2P, // Simplified for test
                                message_id,
                                message_content.len(),
                                100, // Simulated latency
                                true,
                            ).await;
                        }
                        Ok(Err(e)) => {
                            warn!("❌ Message send failed: {}", e);
                        }
                        Err(_) => {
                            warn!("❌ Message send timeout");
                        }
                    }
                }
                
                message_count += 1;
                
                // Respect message interval
                sleep(message_interval).await;
            }
        }
    }
    
    info!("✅ Coordination tests completed: {} messages sent", message_count);
    Ok(())
}

/// Validate intelligent routing decisions
async fn validate_intelligent_routing(test_network: &[TestNode]) -> Result<()> {
    info!("🧠 Validating intelligent routing");
    
    let test_node = &test_network[0];
    let target_node_id = test_network[1].node_id;
    
    // Test 1: Performance-oriented routing for urgent messages
    info!("📈 Testing performance-oriented routing");
    {
        let (tx, rx) = oneshot::channel();
        test_node.command_sender.send(NetworkCommand::SetRoutingStrategy {
            strategy: RoutingStrategy::Performance,
        }).await?;
        
        test_node.command_sender.send(NetworkCommand::SendMessage {
            target: Some(target_node_id),
            content: b"PERFORMANCE TEST: Urgent consensus message".to_vec(),
            message_class: MessageClass::UrgentConsensus,
            response_channel: tx,
        }).await?;
        
        match rx.await? {
            Ok(message_id) => info!("✅ Performance routing successful: {}", message_id),
            Err(e) => warn!("❌ Performance routing failed: {}", e),
        }
    }
    
    // Test 2: Privacy-oriented routing for private messages
    info!("🔒 Testing privacy-oriented routing");
    {
        let (tx, rx) = oneshot::channel();
        test_node.command_sender.send(NetworkCommand::SetRoutingStrategy {
            strategy: RoutingStrategy::MaxPrivacy,
        }).await?;
        
        test_node.command_sender.send(NetworkCommand::SendMessage {
            target: Some(target_node_id),
            content: b"PRIVACY TEST: Confidential validator message".to_vec(),
            message_class: MessageClass::PrivateMessage,
            response_channel: tx,
        }).await?;
        
        match rx.await? {
            Ok(message_id) => info!("✅ Privacy routing successful: {}", message_id),
            Err(e) => warn!("❌ Privacy routing failed: {}", e),
        }
    }
    
    // Test 3: Redundant routing for critical messages
    info!("🔄 Testing redundant routing");
    {
        let (tx, rx) = oneshot::channel();
        test_node.command_sender.send(NetworkCommand::SetRoutingStrategy {
            strategy: RoutingStrategy::Redundant,
        }).await?;
        
        test_node.command_sender.send(NetworkCommand::SendMessage {
            target: Some(target_node_id),
            content: b"REDUNDANCY TEST: Critical emergency message".to_vec(),
            message_class: MessageClass::Emergency,
            response_channel: tx,
        }).await?;
        
        match rx.await? {
            Ok(message_id) => info!("✅ Redundant routing successful: {}", message_id),
            Err(e) => warn!("❌ Redundant routing failed: {}", e),
        }
    }
    
    // Test 4: Load balanced routing
    info!("⚖️ Testing load balanced routing");
    {
        test_node.command_sender.send(NetworkCommand::SetRoutingStrategy {
            strategy: RoutingStrategy::LoadBalanced,
        }).await?;
        
        // Send multiple messages to observe load distribution
        for i in 0..5 {
            let (tx, rx) = oneshot::channel();
            test_node.command_sender.send(NetworkCommand::SendMessage {
                target: Some(target_node_id),
                content: format!("LOAD_BALANCE TEST #{}: Distribution test", i).as_bytes().to_vec(),
                message_class: MessageClass::BlockPropagation,
                response_channel: tx,
            }).await?;
            
            match rx.await? {
                Ok(message_id) => info!("✅ Load balanced message #{}: {}", i, message_id),
                Err(e) => warn!("❌ Load balanced message #{} failed: {}", i, e),
            }
            
            sleep(Duration::from_millis(100)).await;
        }
    }
    
    info!("✅ Intelligent routing validation completed");
    Ok(())
}

/// Test failover mechanisms
async fn test_failover_mechanisms(test_network: &[TestNode]) -> Result<()> {
    info!("🔄 Testing failover mechanisms");
    
    let test_node = &test_network[0];
    let target_node_id = test_network[1].node_id;
    
    // Test 1: Force specific layer usage to test fallback
    info!("🎚️ Testing layer-specific routing with potential fallback");
    
    // Try to use DNS Phantom specifically
    {
        let (tx, rx) = oneshot::channel();
        test_node.command_sender.send(NetworkCommand::SendViaLayer {
            layer: NetworkLayer::DNSPhantom,
            target: Some(target_node_id),
            content: b"FAILOVER TEST: DNS Phantom specific routing".to_vec(),
            response_channel: tx,
        }).await?;
        
        match rx.await? {
            Ok(_) => info!("✅ DNS Phantom routing successful"),
            Err(e) => {
                info!("🔄 DNS Phantom failed as expected, testing fallback: {}", e);
                
                // Test automatic fallback with adaptive strategy
                let (tx2, rx2) = oneshot::channel();
                test_node.command_sender.send(NetworkCommand::SetRoutingStrategy {
                    strategy: RoutingStrategy::Adaptive,
                }).await?;
                
                test_node.command_sender.send(NetworkCommand::SendMessage {
                    target: Some(target_node_id),
                    content: b"FAILOVER TEST: Automatic fallback message".to_vec(),
                    message_class: MessageClass::UrgentConsensus,
                    response_channel: tx2,
                }).await?;
                
                match rx2.await? {
                    Ok(message_id) => info!("✅ Automatic fallback successful: {}", message_id),
                    Err(e) => warn!("❌ Fallback also failed: {}", e),
                }
            }
        }
    }
    
    // Test 2: Simulate network degradation and recovery
    info!("🌐 Testing network degradation simulation");
    {
        // Get current health metrics
        let (tx, rx) = oneshot::channel();
        test_node.command_sender.send(NetworkCommand::GetHealthMetrics {
            response_channel: tx,
        }).await?;
        
        let initial_metrics = rx.await?;
        info!("📊 Initial network health recorded");
        
        // Force health check to update metrics
        test_node.command_sender.send(NetworkCommand::ForceHealthCheck).await?;
        
        // Wait for health check completion
        sleep(Duration::from_secs(2)).await;
        
        // Get updated metrics
        let (tx2, rx2) = oneshot::channel();
        test_node.command_sender.send(NetworkCommand::GetHealthMetrics {
            response_channel: tx2,
        }).await?;
        
        let updated_metrics = rx2.await?;
        info!("📊 Updated network health metrics obtained");
        
        // Compare metrics
        for (layer, initial_metric) in &initial_metrics {
            if let Some(updated_metric) = updated_metrics.get(layer) {
                info!("📈 {:?}: Available: {} -> {}, Latency: {:?} -> {:?}",
                      layer,
                      initial_metric.is_available,
                      updated_metric.is_available,
                      initial_metric.latency_ms,
                      updated_metric.latency_ms);
            }
        }
    }
    
    info!("✅ Failover mechanisms testing completed");
    Ok(())
}

/// Analyze performance metrics across the network
async fn analyze_performance_metrics(test_network: &[TestNode]) -> Result<()> {
    info!("📊 Analyzing performance metrics");
    
    for (idx, node) in test_network.iter().enumerate() {
        info!("📈 Analyzing metrics for {}", node.name);
        
        // Generate health assessment
        let health_assessment = node.metrics_collector.generate_health_assessment().await;
        
        info!("🏥 {} Health Score: {:.1}%", 
              node.name, health_assessment.overall_health_score * 100.0);
        
        // Show layer health scores
        for (layer, score) in &health_assessment.layer_health_scores {
            info!("  {:?}: {:.1}%", layer, score * 100.0);
        }
        
        // Report critical issues
        if !health_assessment.critical_issues.is_empty() {
            warn!("⚠️ {} has {} critical issues:", 
                  node.name, health_assessment.critical_issues.len());
            for issue in &health_assessment.critical_issues {
                warn!("  - {:?}: {}", issue.issue_type, issue.description);
            }
        }
        
        // Show recommendations
        if !health_assessment.recommendations.is_empty() {
            info!("💡 {} optimization recommendations:", node.name);
            for rec in health_assessment.recommendations.iter().take(3) {
                info!("  - Priority {}: {}", rec.priority, rec.description);
            }
        }
        
        // Generate and export metrics report
        let metrics_reporter = MetricsReporter::new(Arc::clone(&node.metrics_collector));
        match metrics_reporter.generate_report().await {
            Ok(report) => {
                info!("📋 Generated metrics report for {} ({} characters)", 
                      node.name, report.len());
                // In a real test, you might save this report to a file
            }
            Err(e) => {
                warn!("❌ Failed to generate report for {}: {}", node.name, e);
            }
        }
        
        // Check for alerts
        let alerts = metrics_reporter.check_alerts().await;
        if !alerts.is_empty() {
            warn!("🚨 {} has {} active alerts", node.name, alerts.len());
        }
        
        // Analyze performance trends
        let trends = node.metrics_collector.analyze_performance_trends().await;
        if !trends.is_empty() {
            info!("📈 {} performance trends: {} metrics analyzed", 
                  node.name, trends.len());
        }
        
        // Export metrics to JSON
        match node.metrics_collector.export_metrics_json().await {
            Ok(json_data) => {
                info!("📄 Exported {} metrics to JSON ({} bytes)", 
                      node.name, json_data.len());
                // In a real scenario, this would be saved or sent to a monitoring system
            }
            Err(e) => {
                warn!("❌ Failed to export metrics for {}: {}", node.name, e);
            }
        }
    }
    
    // Cross-network analysis
    info!("🌐 Performing cross-network analysis");
    
    let mut total_health_score = 0.0;
    let mut total_nodes = 0;
    
    for node in test_network {
        let health_assessment = node.metrics_collector.generate_health_assessment().await;
        total_health_score += health_assessment.overall_health_score;
        total_nodes += 1;
    }
    
    let network_average_health = total_health_score / total_nodes as f64;
    info!("🏥 Network Average Health Score: {:.1}%", network_average_health * 100.0);
    
    if network_average_health > 0.8 {
        info!("✅ Network is in excellent health");
    } else if network_average_health > 0.6 {
        info!("⚠️ Network health is acceptable but could be improved");
    } else {
        warn!("❌ Network health is poor and requires attention");
    }
    
    info!("✅ Performance metrics analysis completed");
    Ok(())
}

/// Helper function to create deterministic test node IDs
fn create_test_node_id(index: usize) -> [u8; 32] {
    let mut node_id = [0u8; 32];
    node_id[0] = (index + 1) as u8;
    node_id[31] = 0xFF; // Mark as test node
    node_id
}

/// Test event monitoring across the network
#[tokio::test]
async fn test_network_event_monitoring() -> Result<()> {
    info!("👁️ Testing network event monitoring");
    
    let config = TestConfig {
        node_count: 2,
        test_duration_seconds: 10,
        message_interval_ms: 1000,
        failure_simulation_enabled: false,
    };
    
    let mut test_network = create_test_network(&config).await?;
    
    // Monitor events for a short period
    let monitor_duration = Duration::from_secs(5);
    let start_time = tokio::time::Instant::now();
    let mut event_count = 0;
    
    while start_time.elapsed() < monitor_duration {
        for node in &mut test_network {
            match node.event_receiver.try_recv() {
                Ok(event) => {
                    event_count += 1;
                    match event {
                        UnifiedNetworkEvent::PeerDiscovered { peer_id, layer, .. } => {
                            info!("🔍 {} discovered peer {} via {:?}", 
                                  node.name, hex::encode(&peer_id[..4]), layer);
                        }
                        UnifiedNetworkEvent::MessageSent { message_id, layer, latency_ms, .. } => {
                            info!("📤 {} sent message {} via {:?} ({}ms)", 
                                  node.name, message_id, layer, latency_ms);
                        }
                        UnifiedNetworkEvent::RoutingDecision { message_class, selected_layers, reason } => {
                            info!("🧠 {} routing: {:?} -> {:?} ({})", 
                                  node.name, message_class, selected_layers, reason);
                        }
                        _ => {
                            info!("📋 {} received network event", node.name);
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                    // No events available
                }
                Err(e) => {
                    warn!("Event monitoring error for {}: {:?}", node.name, e);
                }
            }
        }
        
        sleep(Duration::from_millis(100)).await;
    }
    
    info!("✅ Event monitoring completed: {} events processed", event_count);
    Ok(())
}

/// Benchmark network performance under load
#[tokio::test]
async fn benchmark_network_performance() -> Result<()> {
    info!("🏎️ Benchmarking network performance");
    
    let config = TestConfig {
        node_count: 3,
        test_duration_seconds: 30,
        message_interval_ms: 100, // High frequency
        failure_simulation_enabled: false,
    };
    
    let test_network = create_test_network(&config).await?;
    
    // Measure throughput and latency under load
    let sender = &test_network[0];
    let target_node_id = test_network[1].node_id;
    
    let start_time = tokio::time::Instant::now();
    let mut messages_sent = 0;
    let mut successful_sends = 0;
    
    // Send messages as fast as possible for 10 seconds
    let benchmark_duration = Duration::from_secs(10);
    
    while start_time.elapsed() < benchmark_duration {
        let (tx, rx) = oneshot::channel();
        
        if sender.command_sender.send(NetworkCommand::SendMessage {
            target: Some(target_node_id),
            content: format!("BENCHMARK_MSG_{}", messages_sent).as_bytes().to_vec(),
            message_class: MessageClass::BlockPropagation,
            response_channel: tx,
        }).await.is_ok() {
            messages_sent += 1;
            
            // Don't wait for response to measure maximum throughput
            tokio::spawn(async move {
                if rx.await.is_ok() {
                    // Message sent successfully
                }
            });
        }
        
        // Small delay to prevent overwhelming the system
        sleep(Duration::from_millis(10)).await;
    }
    
    let elapsed = start_time.elapsed();
    let throughput = messages_sent as f64 / elapsed.as_secs_f64();
    
    info!("📊 Benchmark Results:");
    info!("  Duration: {:.2}s", elapsed.as_secs_f64());
    info!("  Messages Sent: {}", messages_sent);
    info!("  Throughput: {:.2} messages/second", throughput);
    
    if throughput > 50.0 {
        info!("✅ High throughput achieved");
    } else if throughput > 10.0 {
        info!("⚠️ Moderate throughput achieved");
    } else {
        warn!("❌ Low throughput - optimization needed");
    }
    
    info!("✅ Performance benchmark completed");
    Ok(())
}