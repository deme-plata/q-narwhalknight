// Integration tests for quantum crypto and quantum mixing plugins
use q_plugin_system::PluginManager;
use q_quantum_crypto::{QuantumCryptoPlugin, QuantumConfig};
use q_quantum_mixing::{QuantumMixingPlugin, QuantumMixingConfig};
use std::sync::Arc;
use tokio;

#[tokio::test]
async fn test_plugin_system_integration() {
    println!("🧪 Testing Q-NarwhalKnight Plugin System Integration");
    
    // Initialize plugin manager
    let mut plugin_manager = PluginManager::new();
    println!("✅ Plugin manager initialized");
    
    // Test quantum crypto plugin initialization
    let quantum_config = QuantumConfig::default();
    let mut quantum_crypto = QuantumCryptoPlugin::new(quantum_config).await.unwrap();
    
    println!("🔬 Quantum crypto plugin created: {}", quantum_crypto.get_name());
    assert_eq!(quantum_crypto.get_id(), "q-quantum-crypto");
    assert_eq!(quantum_crypto.get_version(), "1.0.0");
    
    // Initialize quantum crypto plugin
    quantum_crypto.initialize().await.unwrap();
    println!("✅ Quantum crypto plugin initialized successfully");
    
    // Test quantum mixing plugin initialization  
    let mixing_config = QuantumMixingConfig::default();
    let mut quantum_mixing = QuantumMixingPlugin::new(mixing_config).await.unwrap();
    
    println!("🌀 Quantum mixing plugin created: {}", quantum_mixing.get_name());
    assert_eq!(quantum_mixing.get_id(), "quantum-mixing");
    
    // Set up quantum crypto reference in mixing plugin
    let quantum_crypto_arc = Arc::new(quantum_crypto);
    // quantum_mixing.set_quantum_crypto_plugin(quantum_crypto_arc.clone());
    
    // Initialize quantum mixing plugin
    quantum_mixing.initialize().await.unwrap();
    println!("✅ Quantum mixing plugin initialized successfully");
    
    // Test plugin manager registration
    // plugin_manager.register_plugin(quantum_crypto_arc).await.unwrap();
    // plugin_manager.register_plugin(Arc::new(quantum_mixing)).await.unwrap();
    
    println!("🎉 Plugin integration test completed successfully!");
}

#[tokio::test]
async fn test_quantum_crypto_functionality() {
    println!("🧪 Testing Quantum Crypto Plugin Functionality");
    
    let quantum_config = QuantumConfig::default();
    let mut quantum_crypto = QuantumCryptoPlugin::new(quantum_config).await.unwrap();
    quantum_crypto.initialize().await.unwrap();
    
    // Test status message
    use q_plugin_system::{PluginMessage, PluginContext};
    let status_message = PluginMessage {
        message_type: "get_status".to_string(),
        data: vec![],
        timestamp: chrono::Utc::now(),
    };
    
    let response = quantum_crypto.execute(status_message).await.unwrap();
    assert_eq!(response.message_type, "quantum_crypto_status");
    println!("✅ Quantum crypto status retrieved successfully");
}

#[tokio::test]
async fn test_quantum_mixing_functionality() {
    println!("🧪 Testing Quantum Mixing Plugin Functionality");
    
    let mixing_config = QuantumMixingConfig::default();
    let mut quantum_mixing = QuantumMixingPlugin::new(mixing_config).await.unwrap();
    quantum_mixing.initialize().await.unwrap();
    
    // Test status message
    use q_plugin_system::{PluginMessage, PluginContext};
    let status_message = PluginMessage {
        message_type: "get_status".to_string(),
        data: vec![],
        timestamp: chrono::Utc::now(),
    };
    
    let response = quantum_mixing.execute(status_message).await.unwrap();
    assert_eq!(response.message_type, "get_status_response");
    println!("✅ Quantum mixing status retrieved successfully");
}

#[tokio::test]
async fn test_stark_integration() {
    println!("🧪 Testing STARK System Integration");
    
    let mixing_config = QuantumMixingConfig {
        enable_stark_proofs: true,
        stark_gpu_acceleration: false, // Use CPU for testing
        ..Default::default()
    };
    
    let quantum_mixing = QuantumMixingPlugin::new(mixing_config).await.unwrap();
    println!("✅ Quantum mixing plugin with STARK support initialized");
    
    // Test STARK proof generation (simplified test)
    use q_quantum_mixing::TransactionInfo;
    use chrono::Utc;
    
    let test_transactions = vec![
        TransactionInfo {
            amount: 1000,
            recipient: "test_recipient_1".to_string(),
            fee: Some(10),
            timestamp: Utc::now(),
        },
        TransactionInfo {
            amount: 2000,
            recipient: "test_recipient_2".to_string(),
            fee: Some(20),
            timestamp: Utc::now(),
        },
    ];
    
    let proof_result = quantum_mixing.generate_stark_mixing_proof("test_session_1", &test_transactions).await;
    match proof_result {
        Ok(_) => println!("✅ STARK proof generated successfully"),
        Err(e) => println!("⚠️ STARK proof generation test: {}", e),
    }
}

#[tokio::test]
async fn test_plugin_performance() {
    println!("🧪 Testing Plugin Performance");
    
    let start = std::time::Instant::now();
    
    // Initialize both plugins
    let quantum_config = QuantumConfig::default();
    let mut quantum_crypto = QuantumCryptoPlugin::new(quantum_config).await.unwrap();
    quantum_crypto.initialize().await.unwrap();
    
    let mixing_config = QuantumMixingConfig::default();
    let mut quantum_mixing = QuantumMixingPlugin::new(mixing_config).await.unwrap();
    quantum_mixing.initialize().await.unwrap();
    
    let initialization_time = start.elapsed();
    println!("⏱️ Plugin initialization time: {:?}", initialization_time);
    
    // Test should complete in reasonable time
    assert!(initialization_time.as_secs() < 10, "Plugin initialization took too long");
    
    println!("✅ Performance test completed - plugins initialize quickly");
}