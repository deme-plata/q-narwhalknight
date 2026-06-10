//! Benchmarks for Recursive Proof Generation
//!
//! Measures performance of key operations:
//! - Poseidon hash computation
//! - Circuit constraint building
//! - Proof generation and verification

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_poseidon_hash(c: &mut Criterion) {
    use q_recursive_proofs::gadgets::poseidon::{PoseidonGadget, PoseidonParams};
    use q_recursive_proofs::ConstraintBuilder;

    let params = PoseidonParams::secure_128(8);
    let gadget = PoseidonGadget::new(params.clone());

    c.bench_function("poseidon_native_hash", |b| {
        let inputs = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
        b.iter(|| {
            gadget.hash_native(black_box(&inputs))
        });
    });

    let mut group = c.benchmark_group("poseidon_circuit");
    for size in [2, 4, 8, 16].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut builder = ConstraintBuilder::new();
                let inputs: Vec<usize> = (0..size).map(|i| builder.allocator.alloc_private()).collect();
                gadget.synthesize(&mut builder, black_box(&inputs))
            });
        });
    }
    group.finish();
}

fn bench_merkle_verification(c: &mut Criterion) {
    use q_recursive_proofs::gadgets::merkle::MerkleTreeGadget;
    use q_recursive_proofs::gadgets::poseidon::PoseidonParams;
    use q_recursive_proofs::ConstraintBuilder;

    let params = PoseidonParams::secure_128(16);

    let mut group = c.benchmark_group("merkle_verification");
    for depth in [10, 20, 32].iter() {
        let gadget = MerkleTreeGadget::new(params.clone(), *depth);

        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, _depth| {
            b.iter(|| {
                gadget.estimate_constraints()
            });
        });
    }
    group.finish();
}

fn bench_epoch_transition_circuit(c: &mut Criterion) {
    use q_recursive_proofs::circuits::epoch_transition::{EpochTransitionCircuit, EpochTransitionConfig};

    let config = EpochTransitionConfig::default();
    let circuit = EpochTransitionCircuit::new(config);

    let mut group = c.benchmark_group("epoch_transition");
    for num_blocks in [100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(num_blocks), num_blocks, |b, &num_blocks| {
            b.iter(|| {
                circuit.estimate_constraints(black_box(num_blocks))
            });
        });
    }
    group.finish();
}

fn bench_constraint_estimation(c: &mut Criterion) {
    use q_recursive_proofs::circuits::epoch_transition::{EpochTransitionCircuit, EpochTransitionConfig};
    use q_recursive_proofs::circuits::bft_signature::{BFTSignatureCircuit, BFTSignatureConfig};
    use q_recursive_proofs::circuits::state_transition::{StateTransitionCircuit, StateTransitionConfig};

    c.bench_function("bft_signature_circuit_estimation", |b| {
        let config = BFTSignatureConfig::default();
        let circuit = BFTSignatureCircuit::new(config);
        b.iter(|| {
            circuit.estimate_constraints()
        });
    });

    c.bench_function("state_transition_circuit_estimation", |b| {
        let config = StateTransitionConfig::default();
        let circuit = StateTransitionCircuit::new(config);
        b.iter(|| {
            circuit.estimate_constraints(black_box(100))
        });
    });
}

criterion_group!(
    benches,
    bench_poseidon_hash,
    bench_merkle_verification,
    bench_epoch_transition_circuit,
    bench_constraint_estimation,
);

criterion_main!(benches);
