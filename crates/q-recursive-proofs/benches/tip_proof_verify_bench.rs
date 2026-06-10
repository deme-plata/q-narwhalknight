//! Benchmarks for `LatticeTipProofV2` verification — measures the
//! structural-only verify path that the wasm verifier currently runs in
//! `q-ivc-verifier-wasm` (anchor binding + monotonicity + chain continuity,
//! NO per-step `LatticeGuardProof::verify`).
//!
//! Produced as part of Path C of `/root/.claude/plans/precious-watching-clock.md`:
//! before optimizing or committing to Phase C cryptography, establish what the
//! current verify path actually costs across realistic chain lengths. The 10ms
//! target from `docs/spec-10ms-verification-2026-05-16.tex` was measured against
//! v1 BLAKE3 (0.3-2ms); v2 structural-only is an entirely separate measurement
//! that hasn't been done.
//!
//! Three benchmarks:
//!   - `bench_deserialize_only` — isolates bincode → LatticeTipProofV2 cost so
//!     we know how much of `verify_proof_bytes` time is deser vs verify.
//!   - `bench_tip_verify_v2_structural` — full `tip_verify_v2()` over 10, 100,
//!     1000-step chains. This is what wallets pay today.
//!   - `bench_serialize_only` — symmetric serialize cost; useful for the
//!     producer side of the proof in the API server.
//!
//! Run: `cargo bench -p q-recursive-proofs --bench tip_proof_verify_bench
//!       --features benchmarks`. Criterion HTML report lands under
//! `target/criterion/`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use q_ivc::recursion::{LatticeStepProof, StepIO};
use q_lattice_guard::{params::SecurityLevel, prover::ProofMetadata, LatticeGuardProof};
use q_recursive_proofs::{
    tip_anchor_v2, tip_extend_v2, tip_verify_v2, LatticeTipProofV2,
};

// ════════════════════════════════════════════════════════════════════════════
// Fixtures — mirror the test helpers in tests/tip_proof_v2_e2e.rs so the bench
// and the e2e test share the same notion of a "well-shaped" proof.
// ════════════════════════════════════════════════════════════════════════════

fn root(seed: u8) -> [u8; 32] {
    let mut r = [0u8; 32];
    for (i, b) in r.iter_mut().enumerate() {
        *b = (seed.wrapping_mul(i as u8 + 1)).wrapping_add(seed);
    }
    r
}

fn dummy_lattice_proof() -> LatticeGuardProof {
    LatticeGuardProof {
        commitments: Vec::new(),
        evaluations: (0, 0, 0),
        product_proofs: Vec::new(),
        transcript_state: [0u8; 32],
        metadata: ProofMetadata {
            num_constraints: 0,
            num_public_inputs: 0,
            security_level: SecurityLevel::PQ128,
            generation_time_ms: 0,
        },
    }
}

fn step(z_in: StepIO, z_out: StepIO) -> LatticeStepProof {
    LatticeStepProof {
        proof: dummy_lattice_proof(),
        z_in: z_in.pack(),
        z_out: z_out.pack(),
        public_input_count: 9,
    }
}

/// Build a well-formed `LatticeTipProofV2` with `n` step extensions
/// from genesis anchor `(0, [0u8; 32])`.
fn build_chain(n: u64) -> LatticeTipProofV2 {
    let mut p = tip_anchor_v2(0, [0u8; 32]);
    let mut prev_state = [0u8; 32];
    for h in 0..n {
        let next = root((h + 1) as u8);
        p = tip_extend_v2(&p, step(StepIO::new(prev_state, h), StepIO::new(next, h + 1)))
            .expect("benchmark chain must extend cleanly");
        prev_state = next;
    }
    p
}

// ════════════════════════════════════════════════════════════════════════════
// Benchmarks
// ════════════════════════════════════════════════════════════════════════════

fn bench_tip_verify_v2_structural(c: &mut Criterion) {
    let mut group = c.benchmark_group("tip_verify_v2_structural");
    for &chain_len in &[10u64, 100, 1000] {
        let proof = build_chain(chain_len);
        group.throughput(Throughput::Elements(chain_len));
        group.bench_with_input(
            BenchmarkId::from_parameter(chain_len),
            &proof,
            |b, proof| {
                b.iter(|| {
                    // Anchor matches what build_chain used.
                    tip_verify_v2(black_box(proof), 0, [0u8; 32])
                });
            },
        );
    }
    group.finish();
}

fn bench_deserialize_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("tip_proof_v2_deserialize");
    for &chain_len in &[10u64, 100, 1000] {
        let proof = build_chain(chain_len);
        let bytes = bincode::serialize(&proof).expect("serialise");
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(chain_len),
            &bytes,
            |b, bytes| {
                b.iter(|| {
                    let _p: LatticeTipProofV2 = bincode::deserialize(black_box(bytes))
                        .expect("deserialise");
                });
            },
        );
    }
    group.finish();
}

fn bench_serialize_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("tip_proof_v2_serialize");
    for &chain_len in &[10u64, 100, 1000] {
        let proof = build_chain(chain_len);
        group.throughput(Throughput::Elements(chain_len));
        group.bench_with_input(
            BenchmarkId::from_parameter(chain_len),
            &proof,
            |b, proof| {
                b.iter(|| {
                    bincode::serialize(black_box(proof)).expect("serialise")
                });
            },
        );
    }
    group.finish();
}

/// End-to-end wallet path: bytes → deser → verify → bool. Matches what
/// `q-ivc-verifier-wasm::verify_proof_bytes()` does today (modulo the small
/// header-equality checks). Reports against the "what a wallet pays per
/// HTTP refresh" mental model.
fn bench_end_to_end_wallet_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("tip_proof_v2_e2e_wallet_path");
    for &chain_len in &[10u64, 100, 1000] {
        let proof = build_chain(chain_len);
        let bytes = bincode::serialize(&proof).expect("serialise");
        let expected_tip_height = proof.tip_height;
        let expected_tip_state = proof.tip_state;
        group.throughput(Throughput::Elements(chain_len));
        group.bench_with_input(
            BenchmarkId::from_parameter(chain_len),
            &bytes,
            |b, bytes| {
                b.iter(|| {
                    let proof: LatticeTipProofV2 = bincode::deserialize(black_box(bytes))
                        .expect("deserialise");
                    debug_assert_eq!(proof.tip_height, expected_tip_height);
                    debug_assert_eq!(proof.tip_state, expected_tip_state);
                    tip_verify_v2(&proof, 0, [0u8; 32]).expect("verify")
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_tip_verify_v2_structural,
    bench_deserialize_only,
    bench_serialize_only,
    bench_end_to_end_wallet_path,
);
criterion_main!(benches);
