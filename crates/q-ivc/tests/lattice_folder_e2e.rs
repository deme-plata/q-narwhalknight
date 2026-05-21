//! Phase B1 `LatticeStepFolder` end-to-end integration tests.
//!
//! Exercises the public API surface that downstream crates
//! (`q-recursive-proofs::tip_proof_v2`) consume. The full
//! prove/verify round-trip is `#[ignore]`'d because real SRS generation
//! costs seconds; the bridge-shape and projection tests run by default.

use ark_bls12_381::Fr;
use ark_ff::PrimeField;
use ark_r1cs_std::prelude::*;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, SynthesisError,
};

use q_ivc::recursion::{
    FolderError, LatticeStepFolder, LatticeStepProof, R1csBridge, StepIO,
};
use q_lattice_guard::SecurityLevel;

/// Trivial test circuit: enforces `a * b == c`. Used to exercise the
/// bridge without paying for the full δ-circuit's R1CS.
struct MulCircuit<F: PrimeField> {
    a: F,
    b: F,
    c: F,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for MulCircuit<F> {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<F>,
    ) -> Result<(), SynthesisError> {
        let a = ark_r1cs_std::fields::fp::FpVar::new_input(cs.clone(), || Ok(self.a))?;
        let b = ark_r1cs_std::fields::fp::FpVar::new_input(cs.clone(), || Ok(self.b))?;
        let c = ark_r1cs_std::fields::fp::FpVar::new_input(cs, || Ok(self.c))?;
        let product = &a * &b;
        product.enforce_equal(&c)
    }
}

#[test]
fn bridge_round_trip_on_satisfying_witness() {
    // Satisfying witness: 3 * 5 = 15. arkworks accepts; bridge yields
    // a non-empty ArithmeticCircuit and non-empty assignment vectors.
    let cs = ConstraintSystem::<Fr>::new_ref();
    MulCircuit {
        a: Fr::from(3u64),
        b: Fr::from(5u64),
        c: Fr::from(15u64),
    }
    .generate_constraints(cs.clone())
    .unwrap();
    assert!(cs.is_satisfied().unwrap(), "fixture must satisfy arkworks first");

    let bridge = R1csBridge::from_constraint_system(cs, 1u64 << 32).unwrap();
    let (arith, witness, public_inputs) = bridge.bridge();

    assert!(arith.num_constraints >= 1, "at least one R1CS constraint expected");
    assert_eq!(public_inputs.len(), arith.num_public_inputs);
    assert_eq!(witness.len(), arith.num_witness);
}

#[test]
fn bridge_round_trip_on_unsatisfying_witness_still_extracts_matrices() {
    // arkworks rejects (3 * 5 != 16) but the bridge still produces the
    // matrices — soundness check happens at the lattice-guard prove
    // step, not at the bridge.
    let cs = ConstraintSystem::<Fr>::new_ref();
    MulCircuit {
        a: Fr::from(3u64),
        b: Fr::from(5u64),
        c: Fr::from(16u64), // wrong
    }
    .generate_constraints(cs.clone())
    .unwrap();
    assert!(!cs.is_satisfied().unwrap(), "fixture must NOT satisfy arkworks");

    let bridge = R1csBridge::from_constraint_system(cs, 1u64 << 32).unwrap();
    let (arith, _, _) = bridge.bridge();
    assert!(
        arith.num_constraints >= 1,
        "bridge must extract matrices even from unsatisfying assignments"
    );
}

#[test]
fn bridge_projection_respects_modulus() {
    // A value > modulus must be mod-reduced. Pick a value > 2^31 and
    // modulus = 2^31; result must be < 2^31.
    let modulus: u64 = 1u64 << 31;
    let value = Fr::from((modulus + 17) as u64);

    let cs = ConstraintSystem::<Fr>::new_ref();
    let _ = ark_r1cs_std::fields::fp::FpVar::new_input(cs.clone(), || Ok(value)).unwrap();
    cs.finalize();

    let bridge = R1csBridge::from_constraint_system(cs, modulus).unwrap();
    let public_inputs = bridge.project_public_inputs();
    // First public input is arkworks' implicit "1" constant; second is
    // our value. Both must lie in [0, modulus).
    for v in &public_inputs {
        assert!(*v < modulus, "projection must mod-reduce values into [0, {modulus})");
    }
}

#[test]
fn bridge_constraint_count_matches_arkworks_count() {
    let cs = ConstraintSystem::<Fr>::new_ref();
    MulCircuit {
        a: Fr::from(7u64),
        b: Fr::from(11u64),
        c: Fr::from(77u64),
    }
    .generate_constraints(cs.clone())
    .unwrap();
    let ark_count = cs.num_constraints();

    let bridge = R1csBridge::from_constraint_system(cs, 1u64 << 32).unwrap();
    let arith = bridge.build_circuit();
    assert_eq!(
        arith.num_constraints, ark_count,
        "bridge must preserve constraint count from arkworks"
    );
}

/// Real prove/verify round-trip with SRS generation. Costs seconds;
/// gated behind `#[ignore]`. Run with:
///   cargo test -p q-ivc --test lattice_folder_e2e -- --ignored
#[test]
#[ignore = "expensive: generates a fresh LatticeGuardSRS — ~seconds at pq128"]
fn folder_prove_verify_honest_round_trip() {
    let folder = LatticeStepFolder::new(SecurityLevel::PQ128, 256).expect("SRS gen");
    let z_in = StepIO::new([0u8; 32], 0);
    let z_out = StepIO::new([1u8; 32], 1);

    let proof = folder
        .prove_step::<Fr, _>(
            MulCircuit {
                a: Fr::from(3u64),
                b: Fr::from(5u64),
                c: Fr::from(15u64),
            },
            z_in,
            z_out,
        )
        .expect("prove must succeed on satisfying witness");

    let verified = folder
        .verify_step::<Fr, _, _>(
            &proof,
            || MulCircuit {
                a: Fr::from(3u64),
                b: Fr::from(5u64),
                c: Fr::from(15u64),
            },
            z_in,
            z_out,
        )
        .expect("verify call must not error");
    assert!(verified, "honest proof must verify");
}

#[test]
#[ignore = "expensive: generates a fresh LatticeGuardSRS"]
fn folder_verify_rejects_mismatched_z_out() {
    let folder = LatticeStepFolder::new(SecurityLevel::PQ128, 256).expect("SRS gen");
    let z_in = StepIO::new([0u8; 32], 0);
    let z_out_honest = StepIO::new([1u8; 32], 1);
    let z_out_dishonest = StepIO::new([2u8; 32], 1);

    let proof = folder
        .prove_step::<Fr, _>(
            MulCircuit {
                a: Fr::from(3u64),
                b: Fr::from(5u64),
                c: Fr::from(15u64),
            },
            z_in,
            z_out_honest,
        )
        .expect("prove");

    let result = folder.verify_step::<Fr, _, _>(
        &proof,
        || MulCircuit {
            a: Fr::from(3u64),
            b: Fr::from(5u64),
            c: Fr::from(15u64),
        },
        z_in,
        z_out_dishonest,
    );

    assert!(
        matches!(result, Err(FolderError::StepIoMismatch)),
        "mismatched z_out must surface as StepIoMismatch, got {result:?}"
    );
}

#[test]
fn lattice_step_proof_unpack_roundtrip() {
    use q_lattice_guard::params::SecurityLevel;
    use q_lattice_guard::prover::ProofMetadata;
    use q_lattice_guard::LatticeGuardProof;

    let inner = LatticeGuardProof {
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
    };

    let proof = LatticeStepProof {
        proof: inner,
        z_in: StepIO::new([7u8; 32], 99).pack(),
        z_out: StepIO::new([8u8; 32], 100).pack(),
        public_input_count: 9,
    };

    assert_eq!(proof.z_in_unpacked().state_root, [7u8; 32]);
    assert_eq!(proof.z_in_unpacked().height, 99);
    assert_eq!(proof.z_out_unpacked().state_root, [8u8; 32]);
    assert_eq!(proof.z_out_unpacked().height, 100);
}
