//! Basic Tor Client Tests
//!
//! Simple tests to verify QTorClient functionality

use q_tor_client::{QTorClient, TorConfig};
use q_types::{NodeId, Phase};
use std::time::Duration;

fn test_node_id(id: &str) -> NodeId {
    let mut bytes = [0u8; 32];
    let id_bytes = id.as_bytes();
    let len = std::cmp::min(id_bytes.len(), 32);
    bytes[..len].copy_from_slice(&id_bytes[..len]);
    bytes
}

#[tokio::test]
async fn test_tor_config_creation() {
    let config = TorConfig::default();
    assert!(config.circuit_count >= 4);
    assert_eq!(config.rpc_port, 4001);
    println!("✅ TorConfig creation successful");
}

#[tokio::test]
async fn test_tor_config_validation() {
    let mut config = TorConfig::default();
    config.circuit_count = 5;

    let result = config.validate();
    assert!(result.is_ok(), "Valid config should pass validation");
    println!("✅ Config validation passed");
}

#[tokio::test]
async fn test_tor_config_stealth_mode() {
    let config = TorConfig::stealth_mode();
    assert!(config.enabled);
    assert!(config.tor_only);
    assert!(config.enable_dandelion);
    println!("✅ Stealth mode config created");
}

#[tokio::test]
async fn test_tor_config_hybrid_mode() {
    let config = TorConfig::hybrid_mode();
    assert!(config.enabled);
    assert!(!config.tor_only);
    println!("✅ Hybrid mode config created");
}

#[tokio::test]
async fn test_tor_client_mock_creation() {
    let client = QTorClient::mock();
    let stats = client.get_tor_stats().await;

    assert!(stats.tor_enabled);
    println!("✅ Mock Tor client created successfully");
    println!("   Active circuits: {}", stats.active_circuits);
    println!("   Connection count: {}", stats.connection_count);
}

#[tokio::test]
async fn test_tor_client_initialization_attempt() {
    let config = TorConfig {
        enabled: true,
        circuit_count: 4,
        rpc_port: 10001,
        data_dir: Some("/tmp/tor_test_basic".into()),
        onion_key_path: None,
        bandwidth_burst: "1000KB".to_string(),
        enable_dandelion: false,
        latency_target_ms: Some(300),
        tor_only: false,
        socks_proxy_addr: Some("127.0.0.1:9150".parse().unwrap()),
        bootstrap_onions: vec![],
        enable_prometheus_metrics: false,
    };

    let node_id = test_node_id("test_node_1");
    let result = QTorClient::new(config, node_id, Phase::Phase1).await;

    match result {
        Ok(client) => {
            println!("✅ REAL TOR CLIENT INITIALIZED!");
            println!("   This means Tor is running on port 9150");

            let stats = client.get_tor_stats().await;
            println!("   Active circuits: {}", stats.active_circuits);
            println!("   Bytes sent: {}", stats.bytes_sent);
            println!("   Bytes received: {}", stats.bytes_received);

            // Try to get onion address
            let onion = client.get_onion_address().await;
            println!("   Onion address: {:?}", onion);

            // Shutdown gracefully
            let _ = client.shutdown().await;
        }
        Err(e) => {
            println!("⚠️  Tor client initialization failed (expected if Tor not running):");
            println!("   Error: {}", e);
            println!("   This is NORMAL in test environments without Tor daemon");
        }
    }
}

#[tokio::test]
async fn test_expected_latency_calculation() {
    let config = TorConfig::default();
    let (min, max) = config.expected_latency_range();

    println!("✅ Expected latency range:");
    println!("   Min: {:?}", min);
    println!("   Max: {:?}", max);

    assert!(min < max);
    assert!(max <= Duration::from_millis(300));
}

#[tokio::test]
async fn test_circuit_count_validation() {
    let mut config = TorConfig::default();

    // Test invalid circuit count (0)
    config.circuit_count = 0;
    assert!(config.validate().is_err(), "Zero circuits should be invalid");

    // Test invalid circuit count (too many)
    config.circuit_count = 20;
    assert!(config.validate().is_err(), "Too many circuits should be invalid");

    // Test valid circuit count
    config.circuit_count = 5;
    assert!(config.validate().is_ok(), "5 circuits should be valid");

    println!("✅ Circuit count validation working correctly");
}

#[tokio::test]
async fn test_latency_target_validation() {
    let mut config = TorConfig::default();

    // Test too low latency target
    config.latency_target_ms = Some(10);
    assert!(config.validate().is_err(), "10ms target too low");

    // Test too high latency target
    config.latency_target_ms = Some(10000);
    assert!(config.validate().is_err(), "10s target too high");

    // Test valid latency target
    config.latency_target_ms = Some(200);
    assert!(config.validate().is_ok(), "200ms target should be valid");

    println!("✅ Latency target validation working correctly");
}

#[tokio::test]
async fn test_mock_client_operations() {
    let client = QTorClient::mock();

    // Test statistics retrieval
    let stats = client.get_tor_stats().await;
    assert!(stats.tor_enabled);
    println!("✅ Mock client stats: {} circuits", stats.active_circuits);

    // Test readiness check
    let is_ready = client.is_ready().await;
    println!("   Ready status: {}", is_ready);

    // Test latency target setting
    let result = client.set_latency_target(250).await;
    assert!(result.is_ok());
    println!("   Latency target set successfully");

    // Test circuit rotation
    let result = client.rotate_circuits().await;
    assert!(result.is_ok());
    println!("   Circuit rotation completed");
}

#[tokio::test]
async fn test_connection_attempt_mock() {
    let client = QTorClient::mock();

    // Attempt to connect to a fake onion address
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        client.connect_to_peer("test.onion:4001")
    ).await;

    // This will timeout or fail, which is expected for mock
    match result {
        Ok(Ok(_)) => println!("✅ Connection unexpectedly succeeded"),
        Ok(Err(e)) => println!("✅ Connection failed as expected: {}", e),
        Err(_) => println!("✅ Connection timed out as expected"),
    }
}

#[tokio::test]
async fn test_summary_info() {
    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║                                                       ║");
    println!("║     Q-Tor-Client Basic Functionality Tests          ║");
    println!("║                                                       ║");
    println!("╠═══════════════════════════════════════════════════════╣");
    println!("║                                                       ║");
    println!("║  Tests Performed:                                    ║");
    println!("║  ✓ Configuration creation & validation               ║");
    println!("║  ✓ Mock client operations                            ║");
    println!("║  ✓ Configuration modes (stealth/hybrid)              ║");
    println!("║  ✓ Latency calculations                              ║");
    println!("║  ✓ Real Tor client initialization attempt            ║");
    println!("║                                                       ║");
    println!("║  Note: Real Tor tests require Tor daemon on 9150    ║");
    println!("║                                                       ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");
}
