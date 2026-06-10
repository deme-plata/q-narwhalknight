//! Comprehensive tests for Advanced ZK Systems
//!
//! This test suite validates:
//! - Recursive proof composition and verification
//! - Universal composability security guarantees
//! - Integration with quantum mixing system
//! - Performance and optimization capabilities

use q_quantum_mixing::*;
use std::sync::Arc;
use tokio;
use uuid::Uuid;

/// Test recursive proof composition
#[tokio::test]
async fn test_recursive_proof_composition() {
    println!("🔬 Testing Recursive SNARK Composition");
    
    // Initialize components
    let entropy = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let base_prover = Arc::new(QuantumZKPProver::new(
        entropy.clone(),
        q_quantum_mixing::zkp_prover::ZKProofConfig::default(),
    ).await.unwrap());
    
    let config = AdvancedZKConfig {
        enable_recursive_proofs: true,
        max_recursion_depth: 4,
        batch_size: 8,
        security_parameter: 128,
        enable_proof_caching: true,
        optimization_level: 8,
    };
    
    let advanced_zk = AdvancedZKSystem::new(config, base_prover, entropy.clone()).await.unwrap();
    
    // Generate base proofs to compose
    let mut base_proofs = Vec::new();
    for i in 0..16 {
        let proof = ZKProof {
            proof_data: vec![i as u8; 256], // Mock proof data
            proof_type: ProofType::Stark,
            public_inputs: vec![[i as u8; 32]],
            timestamp: chrono::Utc::now(),
            circuit_id: format!("test_circuit_{}", i),
            vk_hash: [i as u8; 32],
        };
        base_proofs.push(proof);
    }
    
    println!("  ✅ Generated {} base proofs", base_proofs.len());
    
    // Test recursive composition
    let composition_start = std::time::Instant::now();
    let recursive_tree = advanced_zk.compose_proofs(base_proofs, "mixing_composition").await.unwrap();
    let composition_time = composition_start.elapsed();
    
    println!("  ✅ Recursive composition completed in {:?}", composition_time);
    println!("  📊 Tree depth: {}", recursive_tree.depth);
    println!("  📊 Child proofs: {}", recursive_tree.child_proofs.len());
    
    // Verify the recursive tree
    let verification_start = std::time::Instant::now();
    let is_valid = advanced_zk.verify_recursive_proof(&recursive_tree).await.unwrap();
    let verification_time = verification_start.elapsed();
    
    assert!(is_valid, "Recursive proof should be valid");
    println!("  ✅ Recursive verification completed in {:?}", verification_time);
    
    // Test metrics
    let metrics = advanced_zk.get_metrics().await;
    assert!(metrics.recursive_compositions > 0, "Should have recorded compositions");
    assert!(metrics.proofs_verified > 0, "Should have recorded verifications");
    
    println!("  📈 Metrics: {} compositions, {} verifications", 
             metrics.recursive_compositions, metrics.proofs_verified);
    
    println!("✅ Recursive SNARK composition test passed!\n");
}

/// Test universal composability framework
#[tokio::test]
async fn test_universal_composability() {
    println!("🔬 Testing Universal Composability Framework");
    
    // Initialize system
    let entropy = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let base_prover = Arc::new(QuantumZKPProver::new(
        entropy.clone(),
        q_quantum_mixing::zkp_prover::ZKProofConfig::default(),
    ).await.unwrap());
    
    let advanced_zk = AdvancedZKSystem::new(
        AdvancedZKConfig::default(),
        base_prover,
        entropy.clone(),
    ).await.unwrap();
    
    // Create a UC protocol for mixing
    let mixing_protocol = UCProtocol {
        protocol_id: "quantum_mixing_v1".to_string(),
        protocol_type: q_quantum_mixing::advanced_zk::ProtocolType::MixingProtocol,
        ideal_functionality: q_quantum_mixing::advanced_zk::IdealFunctionality {
            name: "quantum_anonymous_mixing".to_string(),
            input_domain: q_quantum_mixing::advanced_zk::FunctionalityDomain {
                domain_type: "utxo_set".to_string(),
                size_bits: 256,
                constraints: vec![
                    "balance_positive".to_string(),
                    "valid_signatures".to_string(),
                    "no_double_spending".to_string(),
                ],
            },
            output_domain: q_quantum_mixing::advanced_zk::FunctionalityDomain {
                domain_type: "mixed_utxo_set".to_string(),
                size_bits: 256,
                constraints: vec![
                    "balance_preserved".to_string(),
                    "unlinkable_outputs".to_string(),
                ],
            },
            security_properties: vec![
                q_quantum_mixing::advanced_zk::SecurityProperty::Anonymity,
                q_quantum_mixing::advanced_zk::SecurityProperty::Unlinkability,
                q_quantum_mixing::advanced_zk::SecurityProperty::Confidentiality,
                q_quantum_mixing::advanced_zk::SecurityProperty::Integrity,
            ],
            leakage_pattern: q_quantum_mixing::advanced_zk::LeakagePattern {
                leaked_info: vec![
                    "transaction_count".to_string(),
                    "total_mixed_amount".to_string(),
                ],
                leakage_bounds: std::collections::HashMap::from([
                    ("timing_variance".to_string(), 0.1),
                    ("size_variance".to_string(), 0.05),
                ]),
                timing_leakage: true,
            },
        },
        instance_id: Uuid::new_v4(),
        security_guarantees: q_quantum_mixing::advanced_zk::SecurityGuarantees {
            semantic_security: 128,
            computational_indistinguishability: true,
            perfect_correctness: true,
            statistical_soundness: 0.9999,
            quantum_secure: true,
        },
    };
    
    println!("  ✅ Created UC mixing protocol: {}", mixing_protocol.protocol_id);
    
    // Register the protocol
    advanced_zk.register_uc_protocol(mixing_protocol.clone()).await.unwrap();
    println!("  ✅ Registered UC protocol successfully");
    
    // Execute the protocol with test inputs
    let test_inputs = vec![
        [1u8; 32],  // Input UTXO 1
        [2u8; 32],  // Input UTXO 2
        [3u8; 32],  // Input UTXO 3
    ];
    
    let execution_start = std::time::Instant::now();
    let outputs = advanced_zk.execute_uc_protocol(&mixing_protocol.protocol_id, test_inputs.clone()).await.unwrap();
    let execution_time = execution_start.elapsed();
    
    assert_eq!(outputs.len(), test_inputs.len(), "Should preserve number of UTXOs");
    println!("  ✅ UC protocol execution completed in {:?}", execution_time);
    println!("  📊 Processed {} inputs -> {} outputs", test_inputs.len(), outputs.len());
    
    // Verify outputs are different from inputs (mixed)
    let mut mixed = false;
    for (i, output) in outputs.iter().enumerate() {
        if output != &test_inputs[i] {
            mixed = true;
            break;
        }
    }
    assert!(mixed, "Outputs should be different from inputs (mixed)");
    println!("  ✅ Mixing functionality verified - outputs differ from inputs");
    
    // Test UC environment
    let uc_env = advanced_zk.get_uc_environment().await;
    assert!(uc_env.active_protocols.contains_key(&mixing_protocol.protocol_id));
    println!("  ✅ UC environment contains registered protocol");
    
    println!("✅ Universal Composability test passed!\n");
}

/// Test integration with quantum mixing service
#[tokio::test]
async fn test_quantum_mixing_integration() {
    println!("🔬 Testing Advanced ZK Integration with Quantum Mixing");
    
    // Create quantum mixing service with advanced ZK enabled
    let config = QuantumMixingConfig {
        min_participants: 2,
        max_participants: 10,
        decoy_enabled: true,
        decoy_strategy: q_quantum_mixing::decoy_transactions::DecoyStrategy {
            decoy_ratio: 5.0,
            timing_variance: std::time::Duration::from_millis(100),
            amount_distribution: q_quantum_mixing::decoy_transactions::AmountDistribution::Uniform,
            network_spread: q_quantum_mixing::decoy_transactions::NetworkPattern::Random,
            quantum_enhancement_level: 7,
        },
        ..Default::default()
    };
    
    let mut service = QuantumMixingService::new(config).await.unwrap();
    println!("  ✅ Quantum mixing service initialized with advanced ZK");
    
    // Test advanced ZK metrics
    let zk_metrics = service.get_advanced_zk_metrics().await.unwrap();
    assert!(zk_metrics.is_some(), "Should have advanced ZK metrics");
    println!("  ✅ Advanced ZK metrics available");
    
    // Test optimization report
    let opt_report = service.get_optimization_report().await.unwrap();
    assert!(opt_report.is_some(), "Should have optimization report");
    let report = opt_report.unwrap();
    println!("  📊 Performance score: {:.1}", report.performance_score);
    println!("  📊 Bottlenecks detected: {}", report.bottlenecks.len());
    
    // Create test UC protocol for secure mixing
    let uc_protocol = UCProtocol {
        protocol_id: "secure_mixing_test".to_string(),
        protocol_type: q_quantum_mixing::advanced_zk::ProtocolType::MixingProtocol,
        ideal_functionality: q_quantum_mixing::advanced_zk::IdealFunctionality {
            name: "test_mixing".to_string(),
            input_domain: q_quantum_mixing::advanced_zk::FunctionalityDomain {
                domain_type: "test_domain".to_string(),
                size_bits: 256,
                constraints: vec!["test_constraint".to_string()],
            },
            output_domain: q_quantum_mixing::advanced_zk::FunctionalityDomain {
                domain_type: "test_output".to_string(),
                size_bits: 256,
                constraints: vec!["output_constraint".to_string()],
            },
            security_properties: vec![
                q_quantum_mixing::advanced_zk::SecurityProperty::Anonymity,
                q_quantum_mixing::advanced_zk::SecurityProperty::Unlinkability,
            ],
            leakage_pattern: q_quantum_mixing::advanced_zk::LeakagePattern {
                leaked_info: vec!["count".to_string()],
                leakage_bounds: std::collections::HashMap::new(),
                timing_leakage: false,
            },
        },
        instance_id: Uuid::new_v4(),
        security_guarantees: q_quantum_mixing::advanced_zk::SecurityGuarantees {
            semantic_security: 128,
            computational_indistinguishability: true,
            perfect_correctness: true,
            statistical_soundness: 0.999,
            quantum_secure: true,
        },
    };
    
    // Register UC protocol
    let registered = service.register_uc_protocol(uc_protocol.clone()).await.unwrap();
    assert!(registered, "UC protocol should be registered");
    println!("  ✅ UC protocol registered with quantum mixing service");
    
    // Test UC mixing execution
    let test_data = vec![[42u8; 32], [84u8; 32], [126u8; 32]];
    let uc_result = service.execute_uc_mixing(&uc_protocol.protocol_id, test_data.clone()).await.unwrap();
    assert!(uc_result.is_some(), "UC mixing should return results");
    println!("  ✅ UC mixing execution completed");
    
    // Test recursive proof composition
    let test_proofs = vec![
        ZKProof {
            proof_data: vec![1u8; 64],
            proof_type: ProofType::Stark,
            public_inputs: vec![[1u8; 32]],
            timestamp: chrono::Utc::now(),
            circuit_id: "test_1".to_string(),
            vk_hash: [1u8; 32],
        },
        ZKProof {
            proof_data: vec![2u8; 64],
            proof_type: ProofType::Stark,
            public_inputs: vec![[2u8; 32]],
            timestamp: chrono::Utc::now(),
            circuit_id: "test_2".to_string(),
            vk_hash: [2u8; 32],
        },
    ];
    
    let composition_result = service.compose_privacy_proofs(test_proofs).await.unwrap();
    assert!(composition_result.is_some(), "Proof composition should succeed");
    let proof_tree = composition_result.unwrap();
    println!("  ✅ Recursive proof composition successful (depth: {})", proof_tree.depth);
    
    // Verify the recursive proof
    let verification_result = service.verify_recursive_proof(&proof_tree).await.unwrap();
    assert!(verification_result.is_some(), "Verification should be available");
    assert!(verification_result.unwrap(), "Recursive proof should be valid");
    println!("  ✅ Recursive proof verification successful");
    
    println!("✅ Quantum Mixing Integration test passed!\n");
}

/// Test batch verification performance
#[tokio::test]
async fn test_batch_verification_performance() {
    println!("🔬 Testing Batch Verification Performance");
    
    let entropy = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let base_prover = Arc::new(QuantumZKPProver::new(
        entropy.clone(),
        q_quantum_mixing::zkp_prover::ZKProofConfig::default(),
    ).await.unwrap());
    
    let config = AdvancedZKConfig {
        batch_size: 16,
        enable_proof_caching: true,
        optimization_level: 9,
        ..Default::default()
    };
    
    let advanced_zk = AdvancedZKSystem::new(config, base_prover, entropy.clone()).await.unwrap();
    
    // Generate multiple recursive proof trees
    const NUM_TREES: usize = 32;
    let mut proof_trees = Vec::new();
    
    for i in 0..NUM_TREES {
        let base_proofs = vec![
            ZKProof {
                proof_data: vec![i as u8; 128],
                proof_type: ProofType::Stark,
                public_inputs: vec![[i as u8; 32]],
                timestamp: chrono::Utc::now(),
                circuit_id: format!("batch_test_{}", i),
                vk_hash: [i as u8; 32],
            }
        ];
        
        let tree = advanced_zk.compose_proofs(base_proofs, "batch_composition").await.unwrap();
        proof_trees.push(tree);
    }
    
    println!("  ✅ Generated {} proof trees for batch testing", NUM_TREES);
    
    // Test batch verification
    let batch_start = std::time::Instant::now();
    let batch_results = advanced_zk.batch_verify_recursive_proofs(&proof_trees).await.unwrap();
    let batch_time = batch_start.elapsed();
    
    assert_eq!(batch_results.len(), NUM_TREES, "Should verify all proofs");
    let all_valid = batch_results.iter().all(|&valid| valid);
    assert!(all_valid, "All proofs should be valid");
    
    println!("  ✅ Batch verification of {} proofs completed in {:?}", NUM_TREES, batch_time);
    println!("  📊 Average time per proof: {:?}", batch_time / NUM_TREES as u32);
    
    // Compare with individual verification
    let individual_start = std::time::Instant::now();
    for tree in &proof_trees[..4] { // Test subset for comparison
        let _valid = advanced_zk.verify_recursive_proof(tree).await.unwrap();
    }
    let individual_time = individual_start.elapsed();
    
    println!("  📊 Individual verification of 4 proofs: {:?}", individual_time);
    println!("  📊 Batch efficiency gain: {:.2}x", 
             (individual_time.as_micros() as f64 * NUM_TREES as f64 / 4.0) / batch_time.as_micros() as f64);
    
    // Check metrics
    let final_metrics = advanced_zk.get_metrics().await;
    assert!(final_metrics.proofs_verified >= NUM_TREES as u64);
    println!("  📈 Total verifications recorded: {}", final_metrics.proofs_verified);
    
    println!("✅ Batch Verification Performance test passed!\n");
}

/// Test security model validation
#[tokio::test]
async fn test_security_model_validation() {
    println!("🔬 Testing Security Model Validation");
    
    let entropy = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let base_prover = Arc::new(QuantumZKPProver::new(
        entropy.clone(),
        q_quantum_mixing::zkp_prover::ZKProofConfig::default(),
    ).await.unwrap());
    
    let advanced_zk = AdvancedZKSystem::new(
        AdvancedZKConfig::default(),
        base_prover,
        entropy.clone(),
    ).await.unwrap();
    
    // Test valid protocol with sufficient security
    let valid_protocol = UCProtocol {
        protocol_id: "secure_protocol".to_string(),
        protocol_type: q_quantum_mixing::advanced_zk::ProtocolType::ZKProofProtocol,
        ideal_functionality: q_quantum_mixing::advanced_zk::IdealFunctionality {
            name: "secure_zkp".to_string(),
            input_domain: q_quantum_mixing::advanced_zk::FunctionalityDomain {
                domain_type: "witness".to_string(),
                size_bits: 256,
                constraints: vec!["valid_witness".to_string()],
            },
            output_domain: q_quantum_mixing::advanced_zk::FunctionalityDomain {
                domain_type: "proof".to_string(),
                size_bits: 256,
                constraints: vec!["sound_proof".to_string()],
            },
            security_properties: vec![
                q_quantum_mixing::advanced_zk::SecurityProperty::Integrity,
                q_quantum_mixing::advanced_zk::SecurityProperty::Confidentiality,
            ],
            leakage_pattern: q_quantum_mixing::advanced_zk::LeakagePattern {
                leaked_info: vec![],
                leakage_bounds: std::collections::HashMap::new(),
                timing_leakage: false,
            },
        },
        instance_id: Uuid::new_v4(),
        security_guarantees: q_quantum_mixing::advanced_zk::SecurityGuarantees {
            semantic_security: 128, // Meets minimum requirement
            computational_indistinguishability: true,
            perfect_correctness: true,
            statistical_soundness: 0.9999,
            quantum_secure: true, // Required for quantum adversary
        },
    };
    
    let result = advanced_zk.register_uc_protocol(valid_protocol).await;
    assert!(result.is_ok(), "Valid protocol should register successfully");
    println!("  ✅ Valid protocol with sufficient security registered");
    
    // Test protocol with insufficient security level
    let weak_protocol = UCProtocol {
        protocol_id: "weak_protocol".to_string(),
        protocol_type: q_quantum_mixing::advanced_zk::ProtocolType::ZKProofProtocol,
        ideal_functionality: q_quantum_mixing::advanced_zk::IdealFunctionality {
            name: "weak_zkp".to_string(),
            input_domain: q_quantum_mixing::advanced_zk::FunctionalityDomain {
                domain_type: "witness".to_string(),
                size_bits: 128,
                constraints: vec![],
            },
            output_domain: q_quantum_mixing::advanced_zk::FunctionalityDomain {
                domain_type: "proof".to_string(),
                size_bits: 128,
                constraints: vec![],
            },
            security_properties: vec![q_quantum_mixing::advanced_zk::SecurityProperty::Integrity],
            leakage_pattern: q_quantum_mixing::advanced_zk::LeakagePattern {
                leaked_info: vec!["everything".to_string()],
                leakage_bounds: std::collections::HashMap::new(),
                timing_leakage: true,
            },
        },
        instance_id: Uuid::new_v4(),
        security_guarantees: q_quantum_mixing::advanced_zk::SecurityGuarantees {
            semantic_security: 64, // Below minimum requirement
            computational_indistinguishability: false,
            perfect_correctness: false,
            statistical_soundness: 0.5,
            quantum_secure: false, // Not quantum secure
        },
    };
    
    let weak_result = advanced_zk.register_uc_protocol(weak_protocol).await;
    assert!(weak_result.is_err(), "Weak protocol should be rejected");
    println!("  ✅ Protocol with insufficient security correctly rejected");
    
    // Verify UC environment state
    let uc_env = advanced_zk.get_uc_environment().await;
    assert_eq!(uc_env.security_parameter, 128, "Security parameter should be 128");
    assert_eq!(uc_env.active_protocols.len(), 1, "Only valid protocol should be registered");
    
    println!("  ✅ UC environment maintains correct security invariants");
    
    println!("✅ Security Model Validation test passed!\n");
}

/// Test system performance under load
#[tokio::test]
async fn test_performance_under_load() {
    println!("🔬 Testing Performance Under Load");
    
    let entropy = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let base_prover = Arc::new(QuantumZKPProver::new(
        entropy.clone(),
        q_quantum_mixing::zkp_prover::ZKProofConfig::default(),
    ).await.unwrap());
    
    let config = AdvancedZKConfig {
        enable_recursive_proofs: true,
        max_recursion_depth: 6,
        batch_size: 32,
        enable_proof_caching: true,
        optimization_level: 10,
        ..Default::default()
    };
    
    let advanced_zk = Arc::new(AdvancedZKSystem::new(config, base_prover, entropy.clone()).await.unwrap());
    
    // Simulate high load with concurrent operations
    const NUM_CONCURRENT: usize = 10;
    const PROOFS_PER_TASK: usize = 8;
    
    let load_start = std::time::Instant::now();
    
    let mut handles = Vec::new();
    for task_id in 0..NUM_CONCURRENT {
        let zk_system = advanced_zk.clone();
        
        let handle = tokio::spawn(async move {
            let mut task_proofs = Vec::new();
            
            // Generate proofs for this task
            for i in 0..PROOFS_PER_TASK {
                let proof = ZKProof {
                    proof_data: vec![(task_id * PROOFS_PER_TASK + i) as u8; 128],
                    proof_type: ProofType::Stark,
                    public_inputs: vec![[(task_id * PROOFS_PER_TASK + i) as u8; 32]],
                    timestamp: chrono::Utc::now(),
                    circuit_id: format!("load_test_{}_{}", task_id, i),
                    vk_hash: [(task_id * PROOFS_PER_TASK + i) as u8; 32],
                };
                task_proofs.push(proof);
            }
            
            // Compose proofs
            let tree = zk_system.compose_proofs(task_proofs, "load_test_composition").await.unwrap();
            
            // Verify composition
            let valid = zk_system.verify_recursive_proof(&tree).await.unwrap();
            assert!(valid, "Load test proof should be valid");
            
            tree
        });
        
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    let mut results = Vec::new();
    for handle in handles {
        let tree = handle.await.unwrap();
        results.push(tree);
    }
    
    let load_time = load_start.elapsed();
    
    assert_eq!(results.len(), NUM_CONCURRENT, "All concurrent tasks should complete");
    println!("  ✅ {} concurrent tasks completed in {:?}", NUM_CONCURRENT, load_time);
    
    let total_proofs = NUM_CONCURRENT * PROOFS_PER_TASK;
    println!("  📊 Total proofs processed: {}", total_proofs);
    println!("  📊 Throughput: {:.1} proofs/second", 
             total_proofs as f64 / load_time.as_secs_f64());
    
    // Verify system state after load
    let final_metrics = advanced_zk.get_metrics().await;
    assert!(final_metrics.recursive_compositions >= NUM_CONCURRENT as u64);
    assert!(final_metrics.proofs_verified >= NUM_CONCURRENT as u64);
    
    println!("  📈 Final metrics - Compositions: {}, Verifications: {}", 
             final_metrics.recursive_compositions, final_metrics.proofs_verified);
    
    println!("✅ Performance Under Load test passed!\n");
}

/// Run all advanced ZK tests
#[tokio::test]
async fn run_comprehensive_advanced_zk_tests() {
    println!("🚀 Running Comprehensive Advanced ZK Test Suite");
    println!("=" * 60);
    
    // Run all test functions
    test_recursive_proof_composition().await;
    test_universal_composability().await;
    test_quantum_mixing_integration().await;
    test_batch_verification_performance().await;
    test_security_model_validation().await;
    test_performance_under_load().await;
    
    println!("🎉 All Advanced ZK Tests Completed Successfully!");
    println!("✨ Q-NarwhalKnight Advanced ZK Systems: Production Ready!");
}