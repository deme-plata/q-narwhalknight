//! # Quantum Mixing Performance Benchmarks
//!
//! Performance benchmarks for the complete quantum mixing system

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use q_quantum_mixing::*;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Benchmark complete mixing flow
fn bench_complete_mixing_flow(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("complete_mixing_3_participants", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = QuantumMixingConfig {
                    min_participants: 2,
                    max_participants: 5,
                    mixing_fee: 10_000,
                    compliance_enabled: false, // Disable for benchmark
                    decoy_enabled: false, // Disable for baseline
                    quantum_enhanced: true,
                };
                
                let mixing_service = QuantumMixingService::new(config).await.unwrap();
                
                // Add 3 participants
                for i in 0..3 {
                    let input = MixingInput {
                        amount: (i + 1) * 1_000_000_000,
                        sender_key: [i as u8; 32],
                        recipient_address: [(i + 10) as u8; 32],
                        commitment: [(i + 20) as u8; 32],
                    };
                    mixing_service.add_participant(input).await.unwrap();
                }
                
                let result = mixing_service.execute_mixing().await.unwrap();
                black_box(result)
            })
        })
    });
}

/// Benchmark with decoy transactions enabled
fn bench_mixing_with_decoys(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("mixing_with_decoys_3_participants", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = QuantumMixingConfig {
                    min_participants: 2,
                    max_participants: 5,
                    mixing_fee: 10_000,
                    compliance_enabled: false,
                    decoy_enabled: true, // Enable decoys
                    quantum_enhanced: true,
                };
                
                let mixing_service = QuantumMixingService::new(config).await.unwrap();
                
                for i in 0..3 {
                    let input = MixingInput {
                        amount: (i + 1) * 1_000_000_000,
                        sender_key: [i as u8; 32],
                        recipient_address: [(i + 10) as u8; 32],
                        commitment: [(i + 20) as u8; 32],
                    };
                    mixing_service.add_participant(input).await.unwrap();
                }
                
                let result = mixing_service.execute_mixing().await.unwrap();
                black_box(result)
            })
        })
    });
}

/// Benchmark quantum entropy generation
fn bench_quantum_entropy(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("quantum_entropy_32_bytes", |b| {
        b.iter(|| {
            rt.block_on(async {
                let entropy_pool = QuantumEntropyPool::new().await.unwrap();
                let mut buffer = [0u8; 32];
                entropy_pool.fill_bytes(&mut buffer).await.unwrap();
                black_box(buffer)
            })
        })
    });
}

criterion_group!(benches, 
    bench_complete_mixing_flow,
    bench_mixing_with_decoys, 
    bench_quantum_entropy
);
criterion_main!(benches);