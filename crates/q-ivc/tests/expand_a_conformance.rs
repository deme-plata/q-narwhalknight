// External conformance + regression scaffolding for FIPS-204 ExpandA.
//
// Follow-up to the byte-order fix landed in commit 0e0b227a4 (DeepSeek
// peer review catch — FIPS-204 §3.2 Algorithm 4 specifies the SHAKE-128
// seed as ρ ∥ i ∥ j, with i the row index and j the column index).
//
// ── What this file does today ──────────────────────────────────────
//
// 1. KAT hash-lock SEEDS are committed (six fixed ρ values covering
//    edge cases: zero, all-ones, sequential, magic, and two patterned
//    seeds). The hash assertions are GATED with `#[ignore]` until an
//    external reference run produces the expected SHA-3-256 digests.
//
// 2. The seed list, serialization format, and assertion harness are
//    locked down. Once any of the three conformance paths in
//    `docs/expand-a-conformance-status.md` produces the expected
//    digests for these seeds, the `#[ignore]` attribute can be lifted
//    in a single follow-up commit and this file becomes permanent
//    conformance coverage.
//
// 3. The `expand_a_print_hashes` helper test (also `#[ignore]`'d so
//    it doesn't run by default) prints our impl's current hashes for
//    every committed seed in copy-paste form. This is the bootstrap
//    step once we add path-A (vendored C reference) or path-C
//    (indirect-via-keygen-reference) cross-checking.
//
// ── What this file deliberately does NOT do ────────────────────────
//
// We do NOT attempt the symmetry-style "byte-order detector" test
// suggested in the original DS-1 draft. Reasoning: under the
// hypothetical swapped-order bug, A[i][j] under buggy = A[j][i]
// under correct — a pure index permutation. The matrix structure
// alone is byte-equivalent up to that permutation, so no internal
// invariant can detect the swap without an external reference.
// The hash-lock with EXTERNALLY-COMPUTED expected digests is the
// only sound conformance check.
//
// See `docs/expand-a-conformance-status.md` for the action plan.

use q_ivc::host::dilithium_witness::{expand_a_native, K, L, N};
use sha3::{Digest, Sha3_256};

/// The six committed seeds. Adding a seed = breaking change for
/// EXPECTED_HASHES. Removing a seed = silently weakening coverage.
fn committed_seeds() -> [(&'static str, [u8; 32]); 6] {
    let mut sequential = [0u8; 32];
    for (i, b) in sequential.iter_mut().enumerate() {
        *b = (i + 1) as u8;
    }
    let mut magic = [0u8; 32];
    magic[..4].copy_from_slice(&[0xCA, 0xFE, 0xBA, 0xBE]);
    [
        ("rho_all_zero",   [0x00u8; 32]),
        ("rho_all_one",    [0x01u8; 32]),
        ("rho_aa",         [0xAAu8; 32]),
        ("rho_all_ff",     [0xFFu8; 32]),
        ("rho_sequential", sequential),
        ("rho_magic",      magic),
    ]
}

/// Row-major u32-little-endian serialization of A. 56 polys × 256 coeffs
/// × 4 bytes = 57 344 bytes.
fn serialize_a_matrix(a: &[[u32; N]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(K * L * N * 4);
    for poly in a {
        for &coeff in poly.iter() {
            out.extend_from_slice(&coeff.to_le_bytes());
        }
    }
    out
}

fn hash_a_matrix(a: &[[u32; N]]) -> [u8; 32] {
    let mut h = Sha3_256::new();
    h.update(&serialize_a_matrix(a));
    h.finalize().into()
}

// ─── KAT hash-lock (gated until we have expected digests) ───────────

/// Once one of the conformance paths in
/// `docs/expand-a-conformance-status.md` produces digests for the six
/// `committed_seeds()`, paste them here in the same order and remove
/// the `#[ignore]` on `expand_a_hash_lock_kat` below.
const EXPECTED_HASHES: &[[u8; 32]; 6] = &[
    [0u8; 32], [0u8; 32], [0u8; 32], [0u8; 32], [0u8; 32], [0u8; 32],
];

#[test]
#[ignore = "Fill in EXPECTED_HASHES from an external reference run, then un-ignore."]
fn expand_a_hash_lock_kat() {
    let seeds = committed_seeds();
    for (idx, (label, rho)) in seeds.iter().enumerate() {
        let a = expand_a_native(rho);
        let h = hash_a_matrix(&a);
        assert_eq!(
            h, EXPECTED_HASHES[idx],
            "ExpandA SHA-3-256 mismatch for {}:\n  got      = 0x{}\n  expected = 0x{}",
            label,
            hex::encode(h),
            hex::encode(EXPECTED_HASHES[idx]),
        );
    }
}

#[test]
#[ignore = "Bootstrap helper: prints our impl's current hashes. Not a check."]
fn expand_a_print_hashes() {
    let seeds = committed_seeds();
    println!("\n// Paste the following into EXPECTED_HASHES:\n");
    for (label, rho) in seeds.iter() {
        let a = expand_a_native(rho);
        let h = hash_a_matrix(&a);
        println!("    /* {:<16} */ {:?},", label, h);
    }
}

// ─── Always-on regression checks (no external reference needed) ─────

#[test]
fn expand_a_matrix_size_and_layout_are_stable() {
    // Locks in: K=8, L=7, N=256, row-major flat Vec layout.
    // Any future refactor that changes constants or layout breaks here.
    let rho = [0x33u8; 32];
    let a = expand_a_native(&rho);
    assert_eq!(K, 8, "FIPS-204 ML-DSA-87: K must be 8");
    assert_eq!(L, 7, "FIPS-204 ML-DSA-87: L must be 7");
    assert_eq!(N, 256, "FIPS-204: ring degree N = 256");
    assert_eq!(a.len(), K * L, "A must be K*L = 56 polynomials");
    for poly in &a {
        assert_eq!(poly.len(), N);
    }
    let bytes = serialize_a_matrix(&a);
    assert_eq!(bytes.len(), K * L * N * 4, "expected 57 344-byte serialization");
}

#[test]
fn expand_a_serialization_is_deterministic() {
    // The hash-lock pattern requires deterministic serialization.
    let rho = [0x99u8; 32];
    let a1 = expand_a_native(&rho);
    let a2 = expand_a_native(&rho);
    let h1 = hash_a_matrix(&a1);
    let h2 = hash_a_matrix(&a2);
    assert_eq!(h1, h2);
}

#[test]
fn expand_a_hash_differs_per_seed() {
    // Sanity: distinct seeds must produce distinct hashes (collision
    // would indicate either a fatal bug or a SHA-3-256 break — both
    // are noteworthy).
    let h_zero = hash_a_matrix(&expand_a_native(&[0u8; 32]));
    let h_one  = hash_a_matrix(&expand_a_native(&[1u8; 32]));
    assert_ne!(h_zero, h_one);
}
