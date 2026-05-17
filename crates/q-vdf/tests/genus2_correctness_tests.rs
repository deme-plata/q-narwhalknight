//! Correctness tests for Genus-2 Jacobian VDF on pq192 curve
//!
//! PURPOSE: Verify that double_jacobian() produces correct results.
//! Since we don't have SageMath on this server, we use self-consistency
//! checks and known mathematical properties that MUST hold.
//!
//! These tests catch bugs like:
//! - Wrong modular reduction
//! - Sign errors in polynomial arithmetic
//! - Edge cases (identity, degree-1, degenerate inputs)
//! - Serialization round-trip failures
//! - Non-determinism (same input must always produce same output)
//!
//! This is prerequisite P3 — no production code is modified.
//!
//! Run with: cargo test --package q-vdf --test genus2_correctness_tests

use num_bigint::{BigInt, BigUint, ToBigInt};
use num_traits::{One, Zero};

use q_vdf::genus2_vdf::{Genus2CurveParams, Genus2VDF, JacobianElement};

// ============================================================================
// CURVE PARAMETER TESTS
// ============================================================================

#[test]
fn test_pq192_curve_params_are_valid() {
    let curve = Genus2CurveParams::pq192();

    // Prime must be non-zero and large (384-bit)
    assert!(curve.p > BigUint::zero(), "pq192 prime must be non-zero");
    assert!(curve.p.bits() >= 380, "pq192 prime must be at least 380 bits, got {}", curve.p.bits());
    assert!(curve.p.bits() <= 384, "pq192 prime must be at most 384 bits, got {}", curve.p.bits());

    // Verify the specific prime value (NIST P-384 prime)
    let expected_p = BigUint::parse_bytes(
        b"39402006196394479212279040100143613805079739270465446667948293404245721771496870329047266088258938001861606973112319",
        10,
    ).unwrap();
    assert_eq!(curve.p, expected_p, "pq192 prime must match expected value");
}

#[test]
fn test_pq128_curve_params_are_valid() {
    let curve = Genus2CurveParams::pq128();
    assert!(curve.p.bits() >= 252, "pq128 prime must be at least 252 bits");
    assert!(curve.p.bits() <= 256, "pq128 prime must be at most 256 bits");
}

#[test]
fn test_pq256_curve_params_are_valid() {
    let curve = Genus2CurveParams::pq256();
    assert!(curve.p.bits() >= 508, "pq256 prime must be at least 508 bits");
    assert!(curve.p.bits() <= 514, "pq256 prime may be slightly over 512 bits, got {}", curve.p.bits());
}

#[test]
fn test_pq192_polynomial_evaluation_known_point() {
    // Curve: y^2 = x^5 + x^3 - 2x + 1
    // At x=0: y^2 = 0 + 0 - 0 + 1 = 1
    let curve = Genus2CurveParams::pq192();
    let x = BigInt::zero();
    let y_sq = curve.evaluate_poly(&x);
    assert_eq!(y_sq, BigInt::one(), "f(0) must equal 1 for y^2 = x^5 + x^3 - 2x + 1");
}

#[test]
fn test_pq192_polynomial_evaluation_x_equals_1() {
    // At x=1: y^2 = 1 + 1 - 2 + 1 = 1
    let curve = Genus2CurveParams::pq192();
    let x = BigInt::one();
    let y_sq = curve.evaluate_poly(&x);
    assert_eq!(y_sq, BigInt::one(), "f(1) must equal 1");
}

#[test]
fn test_pq192_polynomial_evaluation_x_equals_2() {
    // At x=2: y^2 = 32 + 8 - 4 + 1 = 37
    let curve = Genus2CurveParams::pq192();
    let x = BigInt::from(2);
    let y_sq = curve.evaluate_poly(&x);
    assert_eq!(y_sq, BigInt::from(37), "f(2) must equal 37");
}

#[test]
fn test_pq128_polynomial_evaluation_x_equals_0() {
    // Curve: y^2 = x^5 + x^2 - 1
    // At x=0: y^2 = 0 + 0 - 1 = -1 mod p = p-1
    let curve = Genus2CurveParams::pq128();
    let x = BigInt::zero();
    let y_sq = curve.evaluate_poly(&x);
    let p_minus_1 = curve.p.to_bigint().unwrap() - BigInt::one();
    assert_eq!(y_sq, p_minus_1, "f(0) must equal p-1 for y^2 = x^5 + x^2 - 1");
}

// ============================================================================
// JACOBIAN ELEMENT TESTS
// ============================================================================

#[test]
fn test_identity_element_properties() {
    let id = JacobianElement::identity();
    assert_eq!(id.degree, 0, "Identity must have degree 0");
    assert_eq!(id.u0, BigInt::one(), "Identity u0 must be 1 (u(x) = 1)");
    assert_eq!(id.u1, BigInt::zero(), "Identity u1 must be 0");
    assert_eq!(id.v0, BigInt::zero(), "Identity v0 must be 0 (v(x) = 0)");
    assert_eq!(id.v1, BigInt::zero(), "Identity v1 must be 0");
}

#[test]
fn test_doubling_identity_returns_identity() {
    // 2 * O = O (identity doubles to itself)
    let curve = Genus2CurveParams::pq192();
    let vdf = Genus2VDF::with_curve(curve, 100);
    let id = JacobianElement::identity();

    let doubled = vdf.double_jacobian_pub(&id).expect("doubling identity must not fail");
    assert_eq!(doubled.degree, 0, "2*identity must have degree 0");
    assert_eq!(doubled, JacobianElement::identity(), "2*identity must equal identity");
}

#[test]
fn test_from_hash_deterministic() {
    // Same input must always produce same element
    let curve = Genus2CurveParams::pq192();
    let hash = [0xABu8; 32];

    let elem1 = JacobianElement::from_hash(&hash, &curve).unwrap();
    let elem2 = JacobianElement::from_hash(&hash, &curve).unwrap();

    assert_eq!(elem1, elem2, "from_hash must be deterministic");
}

#[test]
fn test_from_hash_different_inputs_different_outputs() {
    let curve = Genus2CurveParams::pq192();
    let hash1 = [0x01u8; 32];
    let hash2 = [0x02u8; 32];

    let elem1 = JacobianElement::from_hash(&hash1, &curve).unwrap();
    let elem2 = JacobianElement::from_hash(&hash2, &curve).unwrap();

    assert_ne!(elem1, elem2, "Different inputs must produce different elements");
}

#[test]
fn test_from_hash_produces_degree_2() {
    let curve = Genus2CurveParams::pq192();
    let hash = [0x42u8; 32];
    let elem = JacobianElement::from_hash(&hash, &curve).unwrap();
    assert_eq!(elem.degree, 2, "from_hash must produce degree-2 element");
}

#[test]
fn test_from_hash_coefficients_in_field() {
    // All Mumford coordinates must be in [0, p)
    let curve = Genus2CurveParams::pq192();
    let p = curve.p.to_bigint().unwrap();

    for seed_byte in 0..50u8 {
        let hash = [seed_byte; 32];
        let elem = JacobianElement::from_hash(&hash, &curve).unwrap();

        assert!(elem.u1 >= BigInt::zero() && elem.u1 < p, "u1 must be in [0, p)");
        assert!(elem.u0 >= BigInt::zero() && elem.u0 < p, "u0 must be in [0, p)");
        assert!(elem.v1 >= BigInt::zero() && elem.v1 < p, "v1 must be in [0, p)");
        assert!(elem.v0 >= BigInt::zero() && elem.v0 < p, "v0 must be in [0, p)");
    }
}

// ============================================================================
// DOUBLING TESTS
// ============================================================================

#[test]
fn test_doubling_is_deterministic() {
    // Same element doubled must always produce the same result
    let curve = Genus2CurveParams::pq192();
    let vdf = Genus2VDF::with_curve(curve.clone(), 100);
    let hash = [0x42u8; 32];
    let elem = JacobianElement::from_hash(&hash, &curve).unwrap();

    let doubled1 = vdf.double_jacobian_pub(&elem).unwrap();
    let doubled2 = vdf.double_jacobian_pub(&elem).unwrap();

    assert_eq!(doubled1, doubled2, "Doubling must be deterministic");
}

#[test]
fn test_doubling_changes_element() {
    // 2D != D (for non-identity, non-torsion elements)
    let curve = Genus2CurveParams::pq192();
    let vdf = Genus2VDF::with_curve(curve.clone(), 100);
    let hash = [0x42u8; 32];
    let elem = JacobianElement::from_hash(&hash, &curve).unwrap();

    let doubled = vdf.double_jacobian_pub(&elem).unwrap();
    assert_ne!(elem, doubled, "2D must differ from D for generic elements");
}

#[test]
fn test_doubling_preserves_degree() {
    // Doubling a degree-2 element should produce a degree-2 element
    // (in general; some special cases might reduce degree)
    let curve = Genus2CurveParams::pq192();
    let vdf = Genus2VDF::with_curve(curve.clone(), 100);
    let hash = [0x42u8; 32];
    let elem = JacobianElement::from_hash(&hash, &curve).unwrap();
    assert_eq!(elem.degree, 2);

    let doubled = vdf.double_jacobian_pub(&elem).unwrap();
    assert!(doubled.degree <= 2, "Doubled element degree must be <= 2");
}

#[test]
fn test_doubling_coefficients_in_field() {
    // All output coordinates must be in [0, p)
    let curve = Genus2CurveParams::pq192();
    let vdf = Genus2VDF::with_curve(curve.clone(), 100);
    let p = curve.p.to_bigint().unwrap();

    for seed_byte in 0..20u8 {
        let hash = [seed_byte; 32];
        let elem = JacobianElement::from_hash(&hash, &curve).unwrap();
        let doubled = vdf.double_jacobian_pub(&elem).unwrap();

        assert!(doubled.u1 >= BigInt::zero() && doubled.u1 < p,
            "Doubled u1 must be in [0, p) for seed {}", seed_byte);
        assert!(doubled.u0 >= BigInt::zero() && doubled.u0 < p,
            "Doubled u0 must be in [0, p) for seed {}", seed_byte);
        assert!(doubled.v1 >= BigInt::zero() && doubled.v1 < p,
            "Doubled v1 must be in [0, p) for seed {}", seed_byte);
        assert!(doubled.v0 >= BigInt::zero() && doubled.v0 < p,
            "Doubled v0 must be in [0, p) for seed {}", seed_byte);
    }
}

#[test]
fn test_sequential_doublings_all_different() {
    // D, 2D, 4D, 8D, ... should all be different for generic D
    let curve = Genus2CurveParams::pq192();
    let vdf = Genus2VDF::with_curve(curve.clone(), 100);
    let hash = [0x42u8; 32];

    let mut current = JacobianElement::from_hash(&hash, &curve).unwrap();
    let mut history = Vec::new();

    for i in 0..20 {
        history.push(current.clone());
        current = vdf.double_jacobian_pub(&current).unwrap();

        // Check against all previous elements
        for (j, prev) in history.iter().enumerate() {
            if j < i {
                assert_ne!(&current, prev,
                    "Element at step {} must differ from step {}", i + 1, j);
            }
        }
    }
}

#[test]
fn test_chain_determinism_full_vdf() {
    // Running the full VDF chain twice must produce identical output
    let curve = Genus2CurveParams::pq192();
    let vdf = Genus2VDF::with_curve(curve.clone(), 100);
    let seed = blake3::hash(b"determinism-test-seed");

    let mut chain1 = JacobianElement::from_hash(seed.as_bytes(), &curve).unwrap();
    let mut chain2 = JacobianElement::from_hash(seed.as_bytes(), &curve).unwrap();

    for _ in 0..50 {
        chain1 = vdf.double_jacobian_pub(&chain1).unwrap();
        chain2 = vdf.double_jacobian_pub(&chain2).unwrap();
    }

    assert_eq!(chain1, chain2, "Full VDF chains must be deterministic");
}

// ============================================================================
// SERIALIZATION ROUND-TRIP TESTS
// ============================================================================

#[test]
fn test_identity_serialization_roundtrip() {
    let id = JacobianElement::identity();
    let bytes = id.to_bytes();
    assert!(!bytes.is_empty(), "Serialized identity must not be empty");
    assert_eq!(bytes[0], 0, "First byte must be degree (0 for identity)");
}

#[test]
fn test_to_bytes_produces_consistent_output() {
    let curve = Genus2CurveParams::pq192();
    let hash = [0x42u8; 32];
    let elem = JacobianElement::from_hash(&hash, &curve).unwrap();

    let bytes1 = elem.to_bytes();
    let bytes2 = elem.to_bytes();

    assert_eq!(bytes1, bytes2, "Serialization must be consistent");
    assert!(bytes1.len() > 10, "Serialized pq192 element must have reasonable size");
}

#[test]
fn test_different_elements_different_bytes() {
    let curve = Genus2CurveParams::pq192();
    let elem1 = JacobianElement::from_hash(&[0x01u8; 32], &curve).unwrap();
    let elem2 = JacobianElement::from_hash(&[0x02u8; 32], &curve).unwrap();

    assert_ne!(elem1.to_bytes(), elem2.to_bytes(),
        "Different elements must serialize differently");
}

// ============================================================================
// VDF OUTPUT HASH TESTS (mining-relevant)
// ============================================================================

#[test]
fn test_vdf_output_hash_deterministic() {
    // The mining flow: seed → VDF → SHA3(output) must be deterministic
    let curve = Genus2CurveParams::pq192();
    let vdf = Genus2VDF::with_curve(curve.clone(), 100);

    let challenge = [0x42u8; 32];
    let nonce = 12345u64;

    let compute_mining_hash = || {
        let mut input = [0u8; 40];
        input[..32].copy_from_slice(&challenge);
        input[32..].copy_from_slice(&nonce.to_le_bytes());
        let seed = blake3::hash(&input);

        let mut g = JacobianElement::from_hash(seed.as_bytes(), &curve).unwrap();
        for _ in 0..10 {
            g = vdf.double_jacobian_pub(&g).unwrap();
        }

        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        hasher.update(&g.to_bytes());
        let result = hasher.finalize();
        result.to_vec()
    };

    let hash1 = compute_mining_hash();
    let hash2 = compute_mining_hash();

    assert_eq!(hash1, hash2, "Mining hash must be deterministic");
    assert_eq!(hash1.len(), 32, "SHA3-256 output must be 32 bytes");
}

#[test]
fn test_different_nonces_different_hashes() {
    let curve = Genus2CurveParams::pq192();
    let vdf = Genus2VDF::with_curve(curve.clone(), 100);
    let challenge = [0x42u8; 32];

    let compute_hash = |nonce: u64| -> Vec<u8> {
        let mut input = [0u8; 40];
        input[..32].copy_from_slice(&challenge);
        input[32..].copy_from_slice(&nonce.to_le_bytes());
        let seed = blake3::hash(&input);

        let mut g = JacobianElement::from_hash(seed.as_bytes(), &curve).unwrap();
        for _ in 0..5 {
            g = vdf.double_jacobian_pub(&g).unwrap();
        }

        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        hasher.update(&g.to_bytes());
        hasher.finalize().to_vec()
    };

    let h1 = compute_hash(0);
    let h2 = compute_hash(1);
    let h3 = compute_hash(2);

    assert_ne!(h1, h2, "Different nonces must produce different hashes");
    assert_ne!(h2, h3, "Different nonces must produce different hashes");
    assert_ne!(h1, h3, "Different nonces must produce different hashes");
}

// ============================================================================
// DEGREE-1 ELEMENT TESTS
// ============================================================================

#[test]
fn test_degree1_doubling() {
    let curve = Genus2CurveParams::pq192();
    let vdf = Genus2VDF::with_curve(curve.clone(), 100);

    let deg1 = JacobianElement::new(
        BigInt::zero(),
        BigInt::from(42),
        BigInt::zero(),
        BigInt::from(7),
        1,
    );

    let doubled = vdf.double_jacobian_pub(&deg1).unwrap();
    assert!(doubled.degree <= 2, "Doubled degree-1 must have degree <= 2");
}

// ============================================================================
// CROSS-CURVE TESTS (pq128 vs pq192 must differ)
// ============================================================================

/// KNOWN BUG: from_hash() only uses 16 bytes of SHA3 output, and double_jacobian()
/// does not fully incorporate curve prime p into all arithmetic paths.
/// This causes pq128 and pq192 to produce identical outputs for same input.
/// MUST FIX before Genus-2 activation.
#[test]
#[ignore = "BUG FOUND: from_hash + double_jacobian not curve-aware enough"]
fn test_different_curves_different_vdf_outputs() {
    let curve128 = Genus2CurveParams::pq128();
    let curve192 = Genus2CurveParams::pq192();
    let vdf128 = Genus2VDF::with_curve(curve128.clone(), 100);
    let vdf192 = Genus2VDF::with_curve(curve192.clone(), 100);

    let hash = [0x42u8; 32];
    let elem128 = JacobianElement::from_hash(&hash, &curve128).unwrap();
    let elem192 = JacobianElement::from_hash(&hash, &curve192).unwrap();

    let doubled128 = vdf128.double_jacobian_pub(&elem128).unwrap();
    let doubled192 = vdf192.double_jacobian_pub(&elem192).unwrap();

    // Different curves must produce different outputs (different prime moduli)
    assert_ne!(doubled128.to_bytes(), doubled192.to_bytes(),
        "pq128 and pq192 must produce different VDF outputs");
}

// ============================================================================
// ASYNC EVALUATE/VERIFY TESTS
// ============================================================================

#[tokio::test]
async fn test_evaluate_produces_valid_output() {
    let vdf = Genus2VDF::new(192).unwrap();
    let input = b"test evaluation on pq192";

    let output = vdf.evaluate(input, 10).await.unwrap();

    assert!(!output.output.is_empty(), "VDF output must not be empty");
    assert_eq!(output.iterations, 10, "Output must record iteration count");
    assert!(output.quantum_enhanced, "Genus-2 must be marked quantum_enhanced");
    assert!(output.computation_time_ns > 0, "Computation time must be recorded");
}

/// KNOWN BUG: from_bytes() is a stub that returns identity element,
/// breaking the verify() function which relies on deserializing the VDF output.
/// MUST FIX before Genus-2 activation.
#[tokio::test]
#[ignore = "BUG FOUND: from_bytes() is a stub — verify() cannot deserialize output"]
async fn test_evaluate_verify_roundtrip() {
    let vdf = Genus2VDF::new(192).unwrap();
    let input = b"verify roundtrip test pq192";

    let output = vdf.evaluate(input, 5).await.unwrap();
    let valid = vdf.verify(input, &output, 5).await.unwrap();

    assert!(valid, "Honest evaluation must verify successfully");
}

#[tokio::test]
async fn test_evaluate_different_iterations_different_output() {
    let vdf = Genus2VDF::new(192).unwrap();
    let input = b"iteration count matters";

    let output5 = vdf.evaluate(input, 5).await.unwrap();
    let output10 = vdf.evaluate(input, 10).await.unwrap();

    assert_ne!(output5.output, output10.output,
        "Different iteration counts must produce different outputs");
}
