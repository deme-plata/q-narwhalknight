//! Comprehensive VM Integration Tests
//! Tests the complete integration of all subsystems

use anyhow::Result;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

// Import VM types and integration modules
use q_vm::integration::{
    coordinator::VMIntegrationCoordinator,
    dag_consensus::{DAGConsensusIntegration, MockCertificate},
    narwhal_broadcast::NarwhalBroadcastIntegration,
    post_quantum::PostQuantumIntegration,
    quantum_vdf::QuantumVDFIntegration,
    CryptographicPhase, IntegrationConfig, VMIntegration,
};
use q_vm::types::VMTransaction;

/// Create test VM transaction
fn create_test_transaction(id: u32) -> VMTransaction {
    VMTransaction {
        id: format!("test-tx-{}", id),
        from: 1000 + id as u64,
        to: 2000 + id as u64,
        value: 100 * id as u64,
        gas_limit: 100000,
        gas_price: 20,
        data: vec![0x60, 0x01, 0x60, 0x02, 0x01], // PUSH1 1 PUSH1 2 ADD
        nonce: id as u64,
        signature: vec![0xaa; 64], // Mock signature
    }
}

/// Create test integration configuration
fn create_test_config() -> IntegrationConfig {
    IntegrationConfig {
        node_id: "integration-test-node".to_string(),
        phase: CryptographicPhase::Phase1,
        enable_quantum_vdf: true,
        vdf_difficulty: 1000,
        mempool_batch_size: 10,
        consensus_timeout_ms: 1000,
    }
}

#[tokio::test]
async fn test_coordinator_initialization() -> Result<()> {
    let config = create_test_config();
    let coordinator = VMIntegrationCoordinator::new(config)?;

    // Initialize all subsystems
    let init_result = timeout(Duration::from_secs(5), coordinator.initialize()).await;

    assert!(
        init_result.is_ok(),
        "Coordinator initialization should complete within timeout"
    );
    assert!(
        init_result.unwrap().is_ok(),
        "Coordinator initialization should succeed"
    );

    // Verify health check
    let health = coordinator.health_check().await;
    assert!(health, "Coordinator should be healthy after initialization");

    println!("✅ Coordinator initialization test passed");
    Ok(())
}

#[tokio::test]
async fn test_single_transaction_processing() -> Result<()> {
    let config = create_test_config();
    let coordinator = VMIntegrationCoordinator::new(config)?;
    coordinator.initialize().await?;

    let tx = create_test_transaction(1);

    // Process transaction through complete pipeline
    let result = timeout(
        Duration::from_secs(10),
        coordinator.process_transaction(tx.clone()),
    )
    .await?;

    assert!(result.is_ok(), "Transaction processing should succeed");
    let integration_result = result.unwrap();

    assert!(
        integration_result.success,
        "Transaction should be processed successfully"
    );
    assert_eq!(integration_result.transaction_hash, tx.id);
    assert!(
        integration_result.processing_time_ms > 0,
        "Processing time should be recorded"
    );
    assert!(
        integration_result.execution_result.gas_used > 0,
        "Gas should be consumed"
    );

    println!("✅ Single transaction processing test passed");
    println!(
        "   Processing time: {}ms",
        integration_result.processing_time_ms
    );
    println!(
        "   Gas used: {}",
        integration_result.execution_result.gas_used
    );
    Ok(())
}

#[tokio::test]
async fn test_batch_transaction_processing() -> Result<()> {
    let config = create_test_config();
    let coordinator = VMIntegrationCoordinator::new(config)?;
    coordinator.initialize().await?;

    // Create batch of transactions
    let transactions: Vec<VMTransaction> = (1..=5).map(create_test_transaction).collect();

    // Process batch
    let results = timeout(
        Duration::from_secs(30),
        coordinator.batch_process(transactions.clone()),
    )
    .await?;

    assert!(results.is_ok(), "Batch processing should succeed");
    let integration_results = results.unwrap();

    assert_eq!(
        integration_results.len(),
        5,
        "All transactions should be processed"
    );

    for (i, result) in integration_results.iter().enumerate() {
        assert!(result.success, "Transaction {} should succeed", i);
        assert_eq!(result.transaction_hash, transactions[i].id);
        assert!(
            result.processing_time_ms > 0,
            "Processing time should be recorded"
        );
    }

    // Check coordinator metrics
    let metrics = coordinator.get_metrics().await;
    assert_eq!(metrics.total_transactions_processed, 5);
    assert_eq!(metrics.successful_integrations, 5);
    assert_eq!(metrics.failed_integrations, 0);
    assert!(metrics.current_tps > 0.0, "TPS should be calculated");

    println!("✅ Batch transaction processing test passed");
    println!(
        "   Transactions processed: {}",
        metrics.total_transactions_processed
    );
    println!("   Current TPS: {:.2}", metrics.current_tps);
    Ok(())
}

#[tokio::test]
async fn test_integrated_system_status() -> Result<()> {
    let config = create_test_config();
    let coordinator = VMIntegrationCoordinator::new(config)?;
    coordinator.initialize().await?;

    // Process a transaction to activate all subsystems
    let tx = create_test_transaction(1);
    coordinator.process_transaction(tx).await?;

    // Get comprehensive system status
    let status = coordinator.get_integrated_status().await?;

    assert!(status.overall_health, "System should be healthy");
    assert!(
        status.dag_consensus.is_healthy,
        "DAG consensus should be healthy"
    );
    assert!(
        status.narwhal_mempool.is_healthy,
        "Narwhal mempool should be healthy"
    );
    assert!(
        status.quantum_vdf.is_healthy,
        "Quantum VDF should be healthy"
    );
    assert!(
        status.post_quantum_crypto.is_healthy,
        "Post-quantum crypto should be healthy"
    );

    // Check metrics
    assert!(status.coordinator_metrics.total_transactions_processed > 0);
    assert!(status.coordinator_metrics.successful_integrations > 0);

    println!("✅ Integrated system status test passed");
    println!("   Overall health: {}", status.overall_health);
    println!(
        "   DAG consensus: {}",
        status.dag_consensus.consensus_status
    );
    println!("   Mempool: {}", status.narwhal_mempool.mempool_status);
    println!("   VDF: {}", status.quantum_vdf.vdf_status);
    println!("   Crypto: {}", status.post_quantum_crypto.crypto_status);
    Ok(())
}

#[tokio::test]
async fn test_subsystem_individual_functionality() -> Result<()> {
    let config = create_test_config();

    // Test DAG Consensus Integration
    let dag_consensus = DAGConsensusIntegration::new("test-node".to_string())?;
    dag_consensus.initialize(&config).await?;

    let tx = create_test_transaction(1);
    let dag_result = dag_consensus.process_transaction(&tx).await?;
    assert!(
        dag_result.success,
        "DAG consensus should process transaction successfully"
    );

    // Test Narwhal Broadcast Integration
    let narwhal_broadcast = NarwhalBroadcastIntegration::new("test-node".to_string());
    narwhal_broadcast.initialize(&config).await?;

    let narwhal_result = narwhal_broadcast.process_transaction(&tx).await?;
    assert!(
        narwhal_result.success,
        "Narwhal broadcast should process transaction successfully"
    );

    // Test Quantum VDF Integration
    let quantum_vdf = QuantumVDFIntegration::new(1000);
    quantum_vdf.initialize(&config).await?;

    let vdf_result = quantum_vdf.process_transaction(&tx).await?;
    assert!(
        vdf_result.success,
        "Quantum VDF should process transaction successfully"
    );
    assert!(vdf_result.vdf_output.is_some(), "VDF should produce output");

    // Test Post-Quantum Cryptography Integration
    let post_quantum = PostQuantumIntegration::new(CryptographicPhase::Phase1)?;
    post_quantum.initialize(&config).await?;

    let crypto_result = post_quantum.process_transaction(&tx).await?;
    assert!(
        crypto_result.success,
        "Post-quantum crypto should process transaction successfully"
    );

    println!("✅ Individual subsystem functionality test passed");
    println!(
        "   DAG consensus gas used: {}",
        dag_result.execution_result.gas_used
    );
    println!(
        "   Narwhal broadcast gas used: {}",
        narwhal_result.execution_result.gas_used
    );
    println!("   VDF gas used: {}", vdf_result.execution_result.gas_used);
    println!(
        "   Crypto gas used: {}",
        crypto_result.execution_result.gas_used
    );
    Ok(())
}

#[tokio::test]
async fn test_cryptographic_phase_transitions() -> Result<()> {
    // Test Phase 0 (Classical)
    let mut config = create_test_config();
    config.phase = CryptographicPhase::Phase0;

    let coordinator_phase0 = VMIntegrationCoordinator::new(config.clone())?;
    coordinator_phase0.initialize().await?;

    let tx = create_test_transaction(1);
    let result_phase0 = coordinator_phase0.process_transaction(tx.clone()).await?;
    assert!(result_phase0.success, "Phase 0 processing should succeed");
    assert_eq!(result_phase0.crypto_phase, CryptographicPhase::Phase0);

    // Test Phase 1 (Hybrid)
    config.phase = CryptographicPhase::Phase1;
    let coordinator_phase1 = VMIntegrationCoordinator::new(config.clone())?;
    coordinator_phase1.initialize().await?;

    let result_phase1 = coordinator_phase1.process_transaction(tx.clone()).await?;
    assert!(result_phase1.success, "Phase 1 processing should succeed");
    assert_eq!(result_phase1.crypto_phase, CryptographicPhase::Phase1);

    // Test Phase 2 (Post-Quantum)
    config.phase = CryptographicPhase::Phase2;
    let coordinator_phase2 = VMIntegrationCoordinator::new(config)?;
    coordinator_phase2.initialize().await?;

    let result_phase2 = coordinator_phase2.process_transaction(tx).await?;
    assert!(result_phase2.success, "Phase 2 processing should succeed");
    assert_eq!(result_phase2.crypto_phase, CryptographicPhase::Phase2);

    println!("✅ Cryptographic phase transitions test passed");
    println!(
        "   Phase 0 gas: {}",
        result_phase0.execution_result.gas_used
    );
    println!(
        "   Phase 1 gas: {}",
        result_phase1.execution_result.gas_used
    );
    println!(
        "   Phase 2 gas: {}",
        result_phase2.execution_result.gas_used
    );
    Ok(())
}

#[tokio::test]
async fn test_performance_benchmarks() -> Result<()> {
    let config = create_test_config();
    let coordinator = VMIntegrationCoordinator::new(config)?;
    coordinator.initialize().await?;

    // Benchmark single transaction processing
    let single_tx = create_test_transaction(1);
    let start_time = std::time::Instant::now();
    let single_result = coordinator.process_transaction(single_tx).await?;
    let single_duration = start_time.elapsed();

    assert!(single_result.success, "Single transaction should succeed");
    assert!(
        single_duration.as_millis() < 5000,
        "Single transaction should complete within 5 seconds"
    );

    // Benchmark batch processing
    let batch_transactions: Vec<VMTransaction> = (1..=10).map(create_test_transaction).collect();

    let batch_start = std::time::Instant::now();
    let batch_results = coordinator.batch_process(batch_transactions).await?;
    let batch_duration = batch_start.elapsed();

    assert_eq!(
        batch_results.len(),
        10,
        "All batch transactions should be processed"
    );
    assert!(
        batch_duration.as_millis() < 15000,
        "Batch processing should complete within 15 seconds"
    );

    // Calculate TPS
    let tps = batch_results.len() as f64 / batch_duration.as_secs_f64();
    assert!(tps > 0.1, "TPS should be reasonable");

    let metrics = coordinator.get_metrics().await;

    println!("✅ Performance benchmarks test passed");
    println!("   Single transaction time: {:?}", single_duration);
    println!("   Batch processing time: {:?}", batch_duration);
    println!("   Calculated TPS: {:.2}", tps);
    println!("   Peak TPS: {:.2}", metrics.peak_tps);
    println!(
        "   Average integration time: {:.2}ms",
        metrics.average_integration_time_ms
    );
    Ok(())
}

#[tokio::test]
async fn test_error_handling_and_recovery() -> Result<()> {
    let config = create_test_config();
    let coordinator = VMIntegrationCoordinator::new(config)?;
    coordinator.initialize().await?;

    // Test invalid transaction (zero gas limit)
    let invalid_tx = VMTransaction {
        gas_limit: 0, // Invalid
        ..create_test_transaction(1)
    };

    let invalid_result = coordinator.process_transaction(invalid_tx).await;
    // Should handle gracefully rather than panicking

    // Test valid transaction after invalid one
    let valid_tx = create_test_transaction(2);
    let valid_result = coordinator.process_transaction(valid_tx).await?;
    assert!(
        valid_result.success,
        "Valid transaction should succeed after invalid one"
    );

    // Test system recovery
    let health_after_error = coordinator.health_check().await;
    assert!(
        health_after_error,
        "System should remain healthy after handling errors"
    );

    println!("✅ Error handling and recovery test passed");
    println!("   System remained healthy after error conditions");
    Ok(())
}

#[tokio::test]
async fn test_concurrent_transaction_processing() -> Result<()> {
    let config = create_test_config();
    let coordinator = Arc::new(VMIntegrationCoordinator::new(config)?);
    coordinator.initialize().await?;

    // Create multiple concurrent transaction processing tasks
    let mut handles = Vec::new();

    for i in 1..=5 {
        let coordinator_clone = coordinator.clone();
        let handle = tokio::spawn(async move {
            let tx = create_test_transaction(i);
            coordinator_clone.process_transaction(tx).await
        });
        handles.push(handle);
    }

    // Wait for all transactions to complete
    let results = futures::future::join_all(handles).await;

    // Verify all transactions succeeded
    for (i, result) in results.into_iter().enumerate() {
        let task_result = result.expect("Task should not panic");
        let integration_result = task_result.expect("Transaction should succeed");
        assert!(
            integration_result.success,
            "Concurrent transaction {} should succeed",
            i + 1
        );
    }

    // Check final metrics
    let metrics = coordinator.get_metrics().await;
    assert_eq!(metrics.total_transactions_processed, 5);
    assert!(
        metrics.current_tps > 0.0,
        "TPS should be calculated for concurrent processing"
    );

    println!("✅ Concurrent transaction processing test passed");
    println!(
        "   Concurrent transactions processed: {}",
        metrics.total_transactions_processed
    );
    println!("   Final TPS: {:.2}", metrics.current_tps);
    Ok(())
}

#[tokio::test]
async fn test_graceful_shutdown() -> Result<()> {
    let config = create_test_config();
    let coordinator = VMIntegrationCoordinator::new(config)?;
    coordinator.initialize().await?;

    // Process a transaction to ensure system is active
    let tx = create_test_transaction(1);
    let result = coordinator.process_transaction(tx).await?;
    assert!(result.success, "Transaction should succeed before shutdown");

    // Test graceful shutdown
    let shutdown_result = timeout(Duration::from_secs(5), coordinator.shutdown()).await;

    assert!(
        shutdown_result.is_ok(),
        "Shutdown should complete within timeout"
    );
    assert!(shutdown_result.unwrap().is_ok(), "Shutdown should succeed");

    println!("✅ Graceful shutdown test passed");
    Ok(())
}

/// Integration test runner
#[tokio::test]
async fn run_all_integration_tests() -> Result<()> {
    println!("🚀 Starting comprehensive VM integration tests...");

    // Run all integration tests in sequence
    test_coordinator_initialization().await?;
    test_single_transaction_processing().await?;
    test_batch_transaction_processing().await?;
    test_integrated_system_status().await?;
    test_subsystem_individual_functionality().await?;
    test_cryptographic_phase_transitions().await?;
    test_performance_benchmarks().await?;
    test_error_handling_and_recovery().await?;
    test_concurrent_transaction_processing().await?;
    test_graceful_shutdown().await?;

    println!("\n🎉 All VM integration tests passed successfully!");
    println!("✨ Q-NarwhalKnight VM integration system is fully operational");
    Ok(())
}
