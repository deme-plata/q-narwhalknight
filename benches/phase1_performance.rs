use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use q_network::{CryptoProvider, Kyber1024KeyExchange, AgileHandshake};
use q_types::Phase;
use libp2p::PeerId;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Performance benchmarks comparing Phase 0 (classical) vs Phase 1 (post-quantum) cryptography
/// Measures key generation, signature/verification, and key exchange performance

fn benchmark_crypto_provider_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("crypto_provider_creation");
    
    group.bench_function("phase0_creation", |b| {
        b.iter(|| {
            black_box(CryptoProvider::new_phase0().expect("Phase 0 provider creation failed"))
        });
    });
    
    group.bench_function("phase1_creation", |b| {
        b.iter(|| {
            black_box(CryptoProvider::new_phase1().expect("Phase 1 provider creation failed"))
        });
    });
    
    group.finish();
}

fn benchmark_kyber1024_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("kyber1024_operations");
    
    // Set longer timeout for key operations
    group.measurement_time(Duration::from_secs(30));
    
    group.bench_function("key_generation", |b| {
        b.to_async(&rt).iter(|| async {
            let mut kex = Kyber1024KeyExchange::new();
            black_box(kex.generate_keypair().await.expect("Key generation failed"))
        });
    });
    
    group.bench_function("key_exchange_full", |b| {
        b.to_async(&rt).iter_with_setup(
            || {
                // Setup: Generate key pairs
                rt.block_on(async {
                    let mut alice_kex = Kyber1024KeyExchange::new();
                    let mut bob_kex = Kyber1024KeyExchange::new();
                    
                    let alice_keys = alice_kex.generate_keypair().await
                        .expect("Alice key generation failed");
                    let bob_keys = bob_kex.generate_keypair().await
                        .expect("Bob key generation failed");
                    
                    (alice_kex, bob_kex, alice_keys, bob_keys)
                })
            },
            |(mut alice_kex, mut bob_kex, alice_keys, bob_keys)| async move {
                // Benchmark: Full key exchange
                let alice_peer = PeerId::random();
                let bob_peer = PeerId::random();
                
                let (alice_secret, ciphertext) = alice_kex.key_exchange(&bob_keys.1, bob_peer).await
                    .expect("Alice key exchange failed");
                let bob_secret = bob_kex.decapsulate(&ciphertext, alice_peer).await
                    .expect("Bob decapsulation failed");
                
                black_box((alice_secret, bob_secret))
            }
        );
    });
    
    group.finish();
}

fn benchmark_scheme_negotiation(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheme_negotiation");
    
    let provider = CryptoProvider::new_phase1().expect("Provider creation failed");
    let peer_schemes = vec![
        q_network::crypto_agile::CryptoScheme {
            signature: q_network::crypto_agile::CryptoSchemeId::Dilithium5,
            kem: q_network::crypto_agile::CryptoSchemeId::Kyber1024,
            hash: q_network::crypto_agile::CryptoSchemeId::SHA3_256,
            vrf: None,
            version: 2,
        },
        q_network::crypto_agile::CryptoScheme {
            signature: q_network::crypto_agile::CryptoSchemeId::Ed25519,
            kem: q_network::crypto_agile::CryptoSchemeId::X25519,
            hash: q_network::crypto_agile::CryptoSchemeId::SHA3_256,
            vrf: None,
            version: 1,
        },
    ];
    
    group.bench_function("negotiate_best_scheme", |b| {
        b.iter(|| {
            black_box(provider.negotiate_scheme(&peer_schemes)
                .expect("Scheme negotiation failed"))
        });
    });
    
    group.finish();
}

fn benchmark_quantum_handshake(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("quantum_handshake");
    
    // Longer timeout for handshake operations
    group.measurement_time(Duration::from_secs(60));
    
    group.bench_function("full_handshake", |b| {
        b.to_async(&rt).iter(|| async {
            let schemes = vec![q_network::crypto_agile::CryptoScheme {
                signature: q_network::crypto_agile::CryptoSchemeId::Dilithium5,
                kem: q_network::crypto_agile::CryptoSchemeId::Kyber1024,
                hash: q_network::crypto_agile::CryptoSchemeId::SHA3_256,
                vrf: None,
                version: 2,
            }];
            
            let mut handshake = AgileHandshake::new(schemes, Phase::Phase1)
                .expect("Handshake creation failed");
            let mut key_exchange = Kyber1024KeyExchange::new();
            let peer_id = PeerId::random();
            
            black_box(handshake.quantum_handshake(peer_id, &mut key_exchange).await
                .expect("Quantum handshake failed"))
        });
    });
    
    group.finish();
}

fn benchmark_crypto_scheme_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("crypto_scheme_comparison");
    
    // Compare Phase 0 vs Phase 1 provider operations
    let phase0_provider = CryptoProvider::new_phase0().expect("Phase 0 provider failed");
    let phase1_provider = CryptoProvider::new_phase1().expect("Phase 1 provider failed");
    
    group.bench_function("phase0_supported_schemes", |b| {
        b.iter(|| {
            black_box(phase0_provider.get_supported_schemes())
        });
    });
    
    group.bench_function("phase1_supported_schemes", |b| {
        b.iter(|| {
            black_box(phase1_provider.get_supported_schemes())
        });
    });
    
    group.bench_function("phase0_capabilities", |b| {
        b.iter(|| {
            black_box(phase0_provider.get_capabilities())
        });
    });
    
    group.bench_function("phase1_capabilities", |b| {
        b.iter(|| {
            black_box(phase1_provider.get_capabilities())
        });
    });
    
    group.finish();
}

fn benchmark_shared_secret_management(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("shared_secret_management");
    
    group.bench_function("secret_storage_retrieval", |b| {
        b.to_async(&rt).iter_with_setup(
            || {
                // Setup: Create key exchange with generated keys
                rt.block_on(async {
                    let mut kex = Kyber1024KeyExchange::new();
                    let keys = kex.generate_keypair().await
                        .expect("Key generation failed");
                    let secret = kex.simulate_key_exchange(&keys.0, &keys.1).await
                        .expect("Secret simulation failed");
                    let peer_id = PeerId::random();
                    
                    // Store secret
                    {
                        let mut secrets = kex.shared_secrets.write().await;
                        secrets.insert(peer_id, secret);
                    }
                    
                    (kex, peer_id)
                })
            },
            |(kex, peer_id)| async move {
                // Benchmark: Retrieve secret
                black_box(kex.get_shared_secret(&peer_id).await)
            }
        );
    });
    
    group.bench_function("secret_cleanup", |b| {
        b.to_async(&rt).iter_with_setup(
            || {
                // Setup: Create key exchange with multiple secrets
                rt.block_on(async {
                    let mut kex = Kyber1024KeyExchange::new();
                    let keys = kex.generate_keypair().await
                        .expect("Key generation failed");
                    
                    // Add multiple secrets
                    for i in 0..10 {
                        let secret = kex.simulate_key_exchange(&keys.0, &keys.1).await
                            .expect("Secret simulation failed");
                        let peer_id = PeerId::random();
                        
                        let mut secrets = kex.shared_secrets.write().await;
                        secrets.insert(peer_id, secret);
                    }
                    
                    kex
                })
            },
            |kex| async move {
                // Benchmark: Cleanup expired secrets
                black_box(kex.cleanup_expired_secrets(0).await
                    .expect("Cleanup failed"))
            }
        );
    });
    
    group.finish();
}

/// Compare performance metrics between Phase 0 and Phase 1
fn performance_comparison_summary() {
    println!("\n📊 Q-NarwhalKnight Phase 1 Performance Summary");
    println!("=============================================");
    println!();
    println!("🎯 Performance Targets:");
    println!("   Phase 0 (Classical):");
    println!("   • Ed25519 signing: ~50µs");
    println!("   • Ed25519 verification: ~150µs"); 
    println!("   • X25519 key exchange: ~50µs");
    println!("   • Network latency: ~12ms");
    println!();
    println!("   Phase 1 (Post-Quantum):");
    println!("   • Dilithium5 signing: <10ms");
    println!("   • Dilithium5 verification: <15ms");
    println!("   • Kyber1024 key generation: <5ms");
    println!("   • Kyber1024 encapsulation: <3ms");
    println!("   • Network latency with PQ: <300ms");
    println!();
    println!("📈 Expected Performance Impact:");
    println!("   • Signature operations: ~200x slower (acceptable for security gain)");
    println!("   • Key exchange: ~100x slower (but still <5ms target)");
    println!("   • Network overhead: ~25x increase in latency");
    println!("   • Memory usage: ~10x increase for signatures");
    println!();
    println!("✅ Trade-offs Analysis:");
    println!("   • Security: Quantum-resistant for 10+ years");
    println!("   • Performance: Acceptable for consensus (still <300ms)");
    println!("   • Scalability: Maintained with batching and SIMD");
    println!("   • Migration: Hybrid mode allows gradual transition");
    println!();
    println!("🚀 Optimization Opportunities:");
    println!("   • SIMD acceleration for lattice operations");
    println!("   • Batch signature verification");
    println!("   • Precomputed key tables");
    println!("   • Hardware acceleration (Phase 3+)");
    println!();
}

criterion_group!(
    benches,
    benchmark_crypto_provider_creation,
    benchmark_kyber1024_operations,
    benchmark_scheme_negotiation,
    benchmark_quantum_handshake,
    benchmark_crypto_scheme_comparison,
    benchmark_shared_secret_management
);

criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_setup() {
        // Verify benchmark setup works
        let _provider = CryptoProvider::new_phase0().expect("Phase 0 provider failed");
        let _provider = CryptoProvider::new_phase1().expect("Phase 1 provider failed");
        println!("✅ Benchmark setup test passed");
    }
    
    #[tokio::test]
    async fn test_performance_baseline() {
        let mut kex = Kyber1024KeyExchange::new();
        
        // Measure key generation time
        let start = std::time::Instant::now();
        let _keys = kex.generate_keypair().await.expect("Key generation failed");
        let duration = start.elapsed();
        
        println!("🚀 Baseline key generation: {:?}", duration);
        
        // For benchmarking, we're more lenient on timing
        assert!(duration.as_secs() < 10, "Key generation took too long");
    }
}