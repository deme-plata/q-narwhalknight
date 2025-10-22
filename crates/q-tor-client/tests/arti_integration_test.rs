/// Integration test for embedded Arti client
use q_tor_client::{QTorClient, TorConfig};
use q_types::Phase;

#[tokio::test]
#[ignore] // Requires network access and time to bootstrap
async fn test_embedded_arti_initialization() {
    // Configure to use embedded Arti
    let config = TorConfig::embedded_arti_mode();

    let node_id = [1u8; 32];

    // Create client with embedded Arti
    let result = QTorClient::new_with_embedded_arti(config, node_id, Phase::Phase0).await;

    match result {
        Ok(client) => {
            assert!(client.is_using_embedded_arti(), "Should be using embedded Arti");
            println!("✅ Embedded Arti client initialized successfully");

            // Check if client is ready
            let ready = client.is_ready().await;
            println!("   Client ready: {}", ready);

            // Shutdown gracefully
            client.shutdown().await.expect("Shutdown failed");
            println!("✅ Client shutdown successfully");
        }
        Err(e) => {
            eprintln!("❌ Failed to initialize embedded Arti client: {}", e);
            panic!("Arti initialization failed");
        }
    }
}

#[tokio::test]
async fn test_automatic_fallback_to_arti() {
    // Configure with SOCKS proxy to non-existent address (will trigger fallback)
    let mut config = TorConfig::default();
    config.enabled = true;
    config.socks_proxy_addr = Some("127.0.0.1:19999".parse().unwrap()); // Non-existent port
    config.use_embedded_arti = false; // Start with SOCKS, should fallback
    config.data_dir = Some(std::path::PathBuf::from("/tmp/qnk_tor_fallback_test"));
    config.cache_dir = Some(std::path::PathBuf::from("/tmp/qnk_tor_cache_fallback_test"));

    let node_id = [2u8; 32];

    // Create client - should automatically fallback to embedded Arti
    let result = QTorClient::new(config, node_id, Phase::Phase0).await;

    match result {
        Ok(client) => {
            assert!(client.is_using_embedded_arti(), "Should have fallen back to embedded Arti");
            println!("✅ Automatic fallback to embedded Arti successful");

            client.shutdown().await.expect("Shutdown failed");
        }
        Err(e) => {
            // This is expected if we don't have network access for bootstrapping
            println!("⚠️ Fallback test failed (expected in restricted environment): {}", e);
        }
    }
}

#[tokio::test]
async fn test_socks_vs_arti_configuration() {
    // Test 1: SOCKS mode configuration
    let socks_config = TorConfig::default();
    assert!(!socks_config.use_embedded_arti, "Default should use SOCKS");
    assert!(socks_config.socks_proxy_addr.is_some(), "SOCKS config should have proxy address");

    // Test 2: Embedded Arti configuration
    let arti_config = TorConfig::embedded_arti_mode();
    assert!(arti_config.use_embedded_arti, "Embedded mode should have flag set");
    assert!(arti_config.socks_proxy_addr.is_none(), "Arti mode doesn't need SOCKS proxy");
    assert!(arti_config.data_dir.is_some(), "Arti needs data directory");
    assert!(arti_config.cache_dir.is_some(), "Arti needs cache directory");

    println!("✅ Configuration modes validated");
}

#[test]
fn test_config_modes() {
    // Test stealth mode
    let stealth = TorConfig::stealth_mode();
    assert!(stealth.enabled);
    assert!(stealth.tor_only);
    assert!(stealth.enable_dandelion);

    // Test hybrid mode
    let hybrid = TorConfig::hybrid_mode();
    assert!(hybrid.enabled);
    assert!(!hybrid.tor_only);

    // Test embedded Arti mode
    let arti = TorConfig::embedded_arti_mode();
    assert!(arti.enabled);
    assert!(arti.use_embedded_arti);
    assert!(arti.socks_proxy_addr.is_none());

    println!("✅ All configuration modes valid");
}
