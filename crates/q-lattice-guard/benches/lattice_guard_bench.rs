//! LatticeGuard Benchmarks
//!
//! Run with: cargo bench --package q-lattice-guard --features benchmarks

#[cfg(feature = "benchmarks")]
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

#[cfg(feature = "benchmarks")]
use q_lattice_guard::{
    ArithmeticCircuit, LatticeGuard, LatticeGuardSRS, RlweParams, SecurityLevel,
};

#[cfg(feature = "benchmarks")]
fn bench_proof_generation(c: &mut Criterion) {
    let params = RlweParams::pq128();
    let mut rng = rand::thread_rng();

    // Generate SRS for various circuit sizes
    let srs = LatticeGuardSRS::generate(params.clone(), 1000, &mut rng)
        .expect("SRS generation should succeed");

    let mut group = c.benchmark_group("proof_generation");

    for num_constraints in [1, 10, 100].iter() {
        let mut circuit = ArithmeticCircuit::new(1, *num_constraints * 2);

        // Add multiplication gates
        for i in 0..*num_constraints {
            circuit.add_multiplication_gate(
                vec![(1 + i * 2, 1)],
                vec![(2 + i * 2, 1)],
                vec![(0, 1)],
            );
        }

        // Create witness and public inputs
        let witness: Vec<u64> = (0..*num_constraints * 2)
            .map(|i| (i as u64 + 2) % 100)
            .collect();
        let public_inputs = vec![witness[0] * witness[1] % params.modulus];

        let lattice_guard = LatticeGuard::new(SecurityLevel::PQ128)
            .expect("LatticeGuard creation should succeed");

        group.bench_with_input(
            BenchmarkId::new("constraints", num_constraints),
            num_constraints,
            |b, _| {
                b.iter(|| {
                    lattice_guard
                        .prove(
                            black_box(&circuit),
                            black_box(&witness),
                            black_box(&public_inputs),
                            black_box(&srs),
                            &mut rand::thread_rng(),
                        )
                        .expect("Proof generation should succeed")
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "benchmarks")]
fn bench_verification(c: &mut Criterion) {
    let params = RlweParams::pq128();
    let mut rng = rand::thread_rng();

    let srs = LatticeGuardSRS::generate(params.clone(), 100, &mut rng)
        .expect("SRS generation should succeed");

    let mut circuit = ArithmeticCircuit::new(1, 2);
    circuit.add_multiplication_gate(vec![(1, 1)], vec![(2, 1)], vec![(0, 1)]);

    let witness = vec![3, 4];
    let public_inputs = vec![12];

    let lattice_guard = LatticeGuard::new(SecurityLevel::PQ128)
        .expect("LatticeGuard creation should succeed");

    let proof = lattice_guard
        .prove(&circuit, &witness, &public_inputs, &srs, &mut rng)
        .expect("Proof generation should succeed");

    c.bench_function("verify_simple_circuit", |b| {
        b.iter(|| {
            lattice_guard
                .verify(
                    black_box(&circuit),
                    black_box(&public_inputs),
                    black_box(&proof),
                    black_box(&srs),
                )
                .expect("Verification should not error")
        })
    });
}

#[cfg(feature = "benchmarks")]
fn bench_ntt_operations(c: &mut Criterion) {
    use q_lattice_guard::ntt::NttOperator;

    let params = RlweParams::pq128();
    let ntt = NttOperator::new(&params);

    let poly: Vec<u64> = (0..params.dimension)
        .map(|i| (i as u64) % params.modulus)
        .collect();

    c.bench_function("ntt_forward", |b| {
        b.iter(|| ntt.forward(black_box(&poly)))
    });

    let ntt_form = ntt.forward(&poly);

    c.bench_function("ntt_inverse", |b| {
        b.iter(|| ntt.inverse(black_box(&ntt_form)))
    });

    c.bench_function("ntt_multiplication", |b| {
        b.iter(|| ntt.mul(black_box(&poly), black_box(&poly)))
    });
}

#[cfg(feature = "benchmarks")]
fn bench_security_levels(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let mut group = c.benchmark_group("security_levels");

    for level in [SecurityLevel::PQ128, SecurityLevel::PQ192].iter() {
        let params = RlweParams::from_security_level(*level);
        let srs = LatticeGuardSRS::generate(params.clone(), 10, &mut rng)
            .expect("SRS generation should succeed");

        let mut circuit = ArithmeticCircuit::new(1, 2);
        circuit.add_multiplication_gate(vec![(1, 1)], vec![(2, 1)], vec![(0, 1)]);

        let witness = vec![3, 4];
        let public_inputs = vec![12];

        let lattice_guard = LatticeGuard::new(*level)
            .expect("LatticeGuard creation should succeed");

        group.bench_with_input(
            BenchmarkId::new("prove", format!("{:?}", level)),
            &level,
            |b, _| {
                b.iter(|| {
                    lattice_guard
                        .prove(
                            black_box(&circuit),
                            black_box(&witness),
                            black_box(&public_inputs),
                            black_box(&srs),
                            &mut rand::thread_rng(),
                        )
                        .expect("Proof generation should succeed")
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "benchmarks")]
criterion_group!(
    benches,
    bench_proof_generation,
    bench_verification,
    bench_ntt_operations,
    bench_security_levels,
);

#[cfg(feature = "benchmarks")]
criterion_main!(benches);

// Dummy main for when benchmarks feature is not enabled
#[cfg(not(feature = "benchmarks"))]
fn main() {
    eprintln!("Run with --features benchmarks to enable benchmarking");
}
