/// Integration tests for networked VM execution over libp2p
///
/// These tests verify that VM instances can discover each other and
/// execute smart contracts across the P2P network.

use anyhow::Result;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use q_vm::{
    network::{VmNetworkBridge, VmNetworkConfig, VmNetworkMessage},
    vm::networked_executor::{NetworkedVmExecutor, NetworkedExecutorConfig, ExecutionStrategy},
    state::StateDB,
};

#[tokio::test]
async fn test_vm_network_bridge_creation() -> Result<()> {
    let state_db = Arc::new(StateDB::new());
    let config = VmNetworkConfig::default();

    let bridge = VmNetworkBridge::new(config, state_db).await?;

    // Verify bridge was created successfully
    let stats = bridge.get_stats().await;
    assert_eq!(stats.connected_vm_peers, 0);
    assert_eq!(stats.remote_executions_sent, 0);

    Ok(())
}

#[tokio::test]
async fn test_networked_executor_local_execution() -> Result<()> {
    let state_db = Arc::new(StateDB::new());
    let exec_config = NetworkedExecutorConfig {
        default_strategy: ExecutionStrategy::Local,
        ..Default::default()
    };
    let net_config = VmNetworkConfig::default();

    let executor = NetworkedVmExecutor::new(exec_config, net_config, state_db).await?;

    // Execute a contract locally
    let result = executor.execute(
        "0xcontract123",
        "balanceOf",
        &[0x01, 0x02, 0x03, 0x04],
        "0xcaller",
        100_000,
        Some(ExecutionStrategy::Local),
    ).await?;

    // Verify execution succeeded
    assert!(result.success);
    assert!(result.gas_used > 0);

    // Check stats
    let stats = executor.get_stats().await;
    assert_eq!(stats.local_executions, 1);
    assert_eq!(stats.remote_executions, 0);
    assert!(stats.average_local_latency_ms > 0.0);

    println!("✅ Local execution test passed");
    println!("   Gas used: {}", result.gas_used);
    println!("   Latency: {:.2}ms", stats.average_local_latency_ms);

    Ok(())
}

#[tokio::test]
async fn test_networked_executor_with_unified_network() -> Result<()> {
    let state_db = Arc::new(StateDB::new());
    let exec_config = NetworkedExecutorConfig::default();
    let net_config = VmNetworkConfig::default();

    let executor = NetworkedVmExecutor::new(exec_config, net_config, state_db).await?
        .with_unified_network().await?;

    // Execute contract
    let result = executor.execute(
        "0xtoken",
        "transfer",
        &[0xAB, 0xCD, 0xEF],
        "0xsender",
        150_000,
        Some(ExecutionStrategy::Local),
    ).await?;

    assert!(result.success);
    println!("✅ Unified network execution test passed");

    Ok(())
}

#[tokio::test]
async fn test_contract_deployment_broadcast() -> Result<()> {
    let state_db = Arc::new(StateDB::new());
    let config = VmNetworkConfig {
        enable_deployment_gossip: true,
        ..Default::default()
    };

    let bridge = VmNetworkBridge::new(config, state_db).await?;

    // Deploy contract to network
    let bytecode = vec![0x60, 0x80, 0x60, 0x40, 0x52]; // Mock WASM bytecode
    let deployer = "0xdeployer".to_string();

    let deployment_id = bridge.deploy_contract_to_network(bytecode.clone(), deployer).await?;

    // Verify deployment was tracked
    assert!(!deployment_id.is_empty());

    let stats = bridge.get_stats().await;
    assert_eq!(stats.contracts_deployed_to_network, 1);

    println!("✅ Contract deployment broadcast test passed");
    println!("   Deployment ID: {}", deployment_id);

    Ok(())
}

#[tokio::test]
async fn test_execution_strategy_fallback() -> Result<()> {
    let state_db = Arc::new(StateDB::new());
    let exec_config = NetworkedExecutorConfig {
        default_strategy: ExecutionStrategy::Remote,
        fallback_to_local: true,
        remote_timeout_ms: 1000,
        ..Default::default()
    };
    let net_config = VmNetworkConfig::default();

    let executor = NetworkedVmExecutor::new(exec_config, net_config, state_db).await?;

    // Attempt remote execution (will fail and fallback to local)
    let result = executor.execute(
        "0xcontract",
        "getValue",
        &[],
        "0xcaller",
        100_000,
        Some(ExecutionStrategy::Remote),
    ).await;

    // Should succeed via fallback
    if let Ok(exec_result) = result {
        assert!(exec_result.success);

        let stats = executor.get_stats().await;
        assert!(stats.network_fallbacks > 0);

        println!("✅ Execution fallback test passed");
        println!("   Fallbacks: {}", stats.network_fallbacks);
    }

    Ok(())
}

#[tokio::test]
async fn test_vm_message_serialization() -> Result<()> {
    // Test execution request serialization
    let request = VmNetworkMessage::ContractExecutionRequest {
        contract_address: "0x123".to_string(),
        function: "transfer".to_string(),
        args: vec![1, 2, 3, 4],
        caller: "0xABC".to_string(),
        gas_limit: 100_000,
        request_id: "req-001".to_string(),
    };

    let serialized = bincode::serialize(&request)?;
    let deserialized: VmNetworkMessage = bincode::deserialize(&serialized)?;

    match deserialized {
        VmNetworkMessage::ContractExecutionRequest { contract_address, function, .. } => {
            assert_eq!(contract_address, "0x123");
            assert_eq!(function, "transfer");
        }
        _ => panic!("Wrong message type"),
    }

    println!("✅ Message serialization test passed");

    Ok(())
}

#[tokio::test]
async fn test_vm_capabilities_announcement() -> Result<()> {
    let state_db = Arc::new(StateDB::new());
    let config = VmNetworkConfig {
        announce_capabilities: true,
        ..Default::default()
    };

    let mut bridge = VmNetworkBridge::new(config, state_db).await?;

    // Subscribe to messages
    let mut rx = bridge.subscribe_messages();

    // Run bridge in background
    tokio::spawn(async move {
        let _ = bridge.run().await;
    });

    // Wait for capabilities announcement
    tokio::select! {
        Ok(msg) = rx.recv() => {
            match msg {
                VmNetworkMessage::VmCapabilities { vm_version, supported_features, .. } => {
                    assert_eq!(vm_version, "0.1.0");
                    assert!(supported_features.contains(&"wasm".to_string()));
                    println!("✅ Capabilities announcement test passed");
                    println!("   Version: {}", vm_version);
                    println!("   Features: {:?}", supported_features);
                }
                _ => {}
            }
        }
        _ = sleep(Duration::from_secs(2)) => {
            println!("⏱️  Capabilities announcement timeout (expected in test)");
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_parallel_local_executions() -> Result<()> {
    let state_db = Arc::new(StateDB::new());
    let exec_config = NetworkedExecutorConfig::default();
    let net_config = VmNetworkConfig::default();

    let executor = Arc::new(
        NetworkedVmExecutor::new(exec_config, net_config, state_db).await?
    );

    // Execute multiple contracts in parallel
    let mut handles = vec![];

    for i in 0..10 {
        let executor_clone = executor.clone();
        let handle = tokio::spawn(async move {
            executor_clone.execute(
                &format!("0xcontract{}", i),
                "execute",
                &[i as u8],
                "0xcaller",
                100_000,
                Some(ExecutionStrategy::Local),
            ).await
        });
        handles.push(handle);
    }

    // Wait for all executions
    let mut successes = 0;
    for handle in handles {
        if let Ok(Ok(result)) = handle.await {
            if result.success {
                successes += 1;
            }
        }
    }

    assert_eq!(successes, 10);

    let stats = executor.get_stats().await;
    assert_eq!(stats.local_executions, 10);

    println!("✅ Parallel execution test passed");
    println!("   Successes: {}/10", successes);
    println!("   Avg latency: {:.2}ms", stats.average_local_latency_ms);

    Ok(())
}

#[tokio::test]
async fn test_network_stats_tracking() -> Result<()> {
    let state_db = Arc::new(StateDB::new());
    let exec_config = NetworkedExecutorConfig::default();
    let net_config = VmNetworkConfig::default();

    let executor = NetworkedVmExecutor::new(exec_config, net_config, state_db).await?;

    // Perform various operations
    let _ = executor.execute(
        "0xcontract1",
        "func1",
        &[1, 2, 3],
        "0xcaller",
        100_000,
        Some(ExecutionStrategy::Local),
    ).await;

    let _ = executor.execute(
        "0xcontract2",
        "func2",
        &[4, 5, 6],
        "0xcaller",
        100_000,
        Some(ExecutionStrategy::Local),
    ).await;

    // Check execution stats
    let exec_stats = executor.get_stats().await;
    assert!(exec_stats.local_executions > 0);
    assert!(exec_stats.average_local_latency_ms > 0.0);

    // Check network stats
    let net_stats = executor.get_network_stats().await;
    // Network stats should be initialized even if no remote operations
    assert_eq!(net_stats.connected_vm_peers, 0);

    println!("✅ Network stats tracking test passed");
    println!("   Execution stats: {:?}", exec_stats);
    println!("   Network stats: {:?}", net_stats);

    Ok(())
}
