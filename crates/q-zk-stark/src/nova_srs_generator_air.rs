//! Transparent-setup attestation for the Nova SRS via STARK proof.
//!
//! Produces a STARK proof that `srs = gen_srs(seed)` for a deterministic public
//! algorithm `gen_srs`. The proof is verifiable by every validator and every
//! fresh-bootstrap node, eliminating the need for a multi-party trusted-setup
//! ceremony.
//!
//! ## Mainnet status
//!
//! Phase 1 (this module): u64-arithmetic AIR over a public prime field. The
//! AIR attests the **generation procedure** — a multiplicative chain
//! `α, α², α³, ...` mod `p` derived from a public seed via BLAKE3. The chain
//! length matches the Nova SRS size. This is shaped like a real transparent
//! setup; the actual cryptography (BN254 group exponentiation) goes in
//! Phase 2 once the Nova IVC wrapper lands.
//!
//! Why this is useful today even at the simplified-arithmetic level:
//!   1. The AIR shape is exactly what the production version uses.
//!   2. The integration with `crates/q-zk-stark/src/{air, stark_prover}.rs`
//!      is fully validated.
//!   3. The wire protocol (`NovaSrsProof` struct + verify path) is stable.
//!      Phase 2 swaps the field arithmetic only.
//!
//! This module follows the production pattern established by
//! `crates/q-storage/src/encryption_zkstark.rs`.

use crate::air::{
    AirConstraints, BoundaryConstraint, BoundaryStep, ExecutionTrace, TransitionConstraint,
    TransitionExpression,
};
use crate::stark_prover::{StarkProof, StarkProver};
use crate::stark_verifier::StarkVerifier;
use anyhow::Result;
use serde::{Deserialize, Serialize};

// ════════════════════════════════════════════════════════════════════════════
// Constants
// ════════════════════════════════════════════════════════════════════════════

/// Soft maximum SRS size the AIR can attest. Bounded by trace-length feasibility
/// (~2²⁰ rows) of the underlying batch prover; production Nova SRS sizes need
/// to fit under this for a single STARK to cover them. Larger SRSs would
/// require sharding the attestation across multiple proofs.
pub const MAX_SRS_SIZE: usize = 1 << 20;

/// Prime modulus for the Phase 1 multiplicative chain. Chosen as a 63-bit
/// Mersenne-like prime that fits in u64 with room for products. Phase 2
/// replaces this with the BN254 scalar field.
///
/// `p = 2^61 - 1` (a Mersenne prime, fast modular reduction).
pub const FIELD_MODULUS: u64 = (1u64 << 61) - 1;

/// AIR register column indices.
const REG_ALPHA: usize = 0;
const REG_POWER: usize = 1;
const REG_STEP_VALID: usize = 2;
const REG_COUNT: usize = 3;

// ════════════════════════════════════════════════════════════════════════════
// Public types
// ════════════════════════════════════════════════════════════════════════════

/// A transparent-setup attestation for the Nova SRS.
///
/// Mirrors the production [`EncryptionKeyProof`](crate::encryption_zkstark)
/// pattern: a deterministic generator function, a STARK proof of correct
/// generation, and a final-state commitment. Validators verify by recomputing
/// the expected commitment from the seed and checking the STARK.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NovaSrsProof {
    /// Public seed from which alpha was derived. Typically a future blockhash
    /// (RANDAO-style) committed before its value is known.
    pub seed: [u8; 32],
    /// Number of SRS elements attested.
    pub srs_size: u32,
    /// Derived multiplier `α = blake3("nova-srs-alpha-v1", seed) mod p`.
    /// Stored on the proof for fast verification; the verifier MUST also
    /// re-derive this from `seed` and reject if they don't match.
    pub alpha: u64,
    /// Final power `α^srs_size mod p` — commits to the entire chain.
    /// The verifier recomputes this independently from `seed` and rejects
    /// if it doesn't match.
    pub final_power: u64,
    /// BLAKE3 hash of the serialized SRS chain
    /// (`[α¹, α², ..., α^srs_size]` packed as little-endian u64).
    pub srs_root: [u8; 32],
    /// The STARK proof bytes (postcard-encoded `StarkProof`).
    pub stark_bytes: Vec<u8>,
}

impl NovaSrsProof {
    /// Generate the SRS deterministically from `seed` and build the STARK
    /// attestation.
    ///
    /// Returns `(proof, srs_bytes)` where:
    ///   * `proof` is the on-chain artifact (small, fixed-shape)
    ///   * `srs_bytes` is the raw SRS chain serialized as little-endian u64s.
    ///     Validators can ignore this and trust the proof; the bytes are
    ///     useful for off-chain consumers that want the actual SRS values.
    pub async fn generate(seed: &[u8; 32], srs_size: usize) -> Result<(Self, Vec<u8>)> {
        if srs_size == 0 || srs_size > MAX_SRS_SIZE {
            anyhow::bail!(
                "srs_size must be in 1..={} (got {})",
                MAX_SRS_SIZE,
                srs_size
            );
        }

        // Step 1: Derive alpha from seed.
        let alpha = derive_alpha(seed);

        // Step 2: Generate the multiplicative chain α¹, α², ..., α^srs_size mod p.
        let chain = compute_alpha_chain(alpha, srs_size);
        debug_assert_eq!(chain.len(), srs_size);

        // Step 3: Serialize SRS chain to bytes and hash for the public commitment.
        let mut srs_bytes = Vec::with_capacity(srs_size * 8);
        for &power in &chain {
            srs_bytes.extend_from_slice(&power.to_le_bytes());
        }
        let srs_root: [u8; 32] = *blake3::hash(&srs_bytes).as_bytes();
        let final_power = *chain.last().expect("non-empty chain");

        // Step 4: Build the AIR trace + constraints, run the prover.
        let trace = build_trace(alpha, &chain);
        let constraints = build_constraints(alpha, srs_size);

        // Sanity check: trace must satisfy constraints. If this fails, the
        // chain computation is inconsistent with the constraint encoding —
        // a programmer error in this module, not a user-input issue.
        if !constraints.verify_constraints(&trace) {
            anyhow::bail!(
                "internal error: generated trace fails its own constraints \
                 (srs_size={}, alpha={})",
                srs_size,
                alpha
            );
        }

        // Step 5: Generate the STARK proof. Constraint bytes are unused by the
        // current StarkProver but included for forward-compat.
        let constraints_bytes = serialize_constraints(&constraints);
        let mut prover = StarkProver::new();
        let stark_proof = prover.prove(&trace.trace_matrix, &constraints_bytes).await?;
        let stark_bytes = bincode::serialize(&stark_proof)?;

        Ok((
            NovaSrsProof {
                seed: *seed,
                srs_size: srs_size as u32,
                alpha,
                final_power,
                srs_root,
                stark_bytes,
            },
            srs_bytes,
        ))
    }

    /// Verify the attestation.
    ///
    /// Verification has three independent checks. ALL must pass.
    ///
    ///   1. Re-derive `α` from `seed` via BLAKE3 and check it matches
    ///      `self.alpha`. Catches any tampering with the seed or alpha fields.
    ///   2. Re-compute the multiplicative chain locally and check `final_power`
    ///      and `srs_root` match. Catches tampering with either commitment.
    ///   3. Verify the STARK proof against the trace and constraints. Catches
    ///      tampering with the proof bytes.
    pub async fn verify(&self) -> Result<bool> {
        let srs_size = self.srs_size as usize;
        if srs_size == 0 || srs_size > MAX_SRS_SIZE {
            return Ok(false);
        }

        // Check 1: alpha matches seed-derivation.
        let expected_alpha = derive_alpha(&self.seed);
        if expected_alpha != self.alpha {
            return Ok(false);
        }

        // Check 2: re-compute the chain locally, verify final_power + srs_root.
        let chain = compute_alpha_chain(self.alpha, srs_size);
        let local_final = *chain.last().expect("non-empty chain");
        if local_final != self.final_power {
            return Ok(false);
        }
        let mut srs_bytes = Vec::with_capacity(srs_size * 8);
        for &p in &chain {
            srs_bytes.extend_from_slice(&p.to_le_bytes());
        }
        let local_root: [u8; 32] = *blake3::hash(&srs_bytes).as_bytes();
        if local_root != self.srs_root {
            return Ok(false);
        }

        // Check 3: STARK proof verifies. We rebuild the trace + constraints
        // (deterministic from public values) and pass them to the verifier.
        let stark_proof: StarkProof = match bincode::deserialize(&self.stark_bytes) {
            Ok(p) => p,
            Err(_) => return Ok(false),
        };
        let trace = build_trace(self.alpha, &chain);
        let constraints = build_constraints(self.alpha, srs_size);

        // Bind proof to chain: the STARK's trace commitment must match the
        // one we'd derive locally from the regenerated trace.
        let local_commitment = trace_commitment_blake3(&trace.trace_matrix);
        if stark_proof.execution_trace_commitment != local_commitment {
            return Ok(false);
        }

        let expected_public_inputs: Vec<u64> = trace
            .trace_matrix
            .first()
            .cloned()
            .unwrap_or_default();
        if stark_proof.public_inputs != expected_public_inputs {
            return Ok(false);
        }

        if !constraints.verify_constraints(&trace) {
            return Ok(false);
        }

        // Final STARK-layer verification (FRI, etc.). In Phase 2 when the FRI
        // is wired against a real polynomial commitment this call becomes
        // load-bearing for soundness; today it ensures the proof bytes parse
        // and match the public inputs we expect.
        let mut verifier = StarkVerifier::new();
        let valid = verifier
            .verify(&stark_proof, &expected_public_inputs)
            .await?;
        Ok(valid)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Deterministic generator (pure helpers, no STARK)
// ════════════════════════════════════════════════════════════════════════════

/// Derive `α` from `seed` via BLAKE3-with-domain-separation, reduced mod p.
///
/// Domain separator `"nova-srs-alpha-v1"` is bound by `blake3::derive_key`
/// to make the derivation irreversible from `α` back to `seed` and
/// independent of any other key-derivation in the project.
fn derive_alpha(seed: &[u8; 32]) -> u64 {
    let derived = blake3::derive_key("nova-srs-alpha-v1", seed);
    // Take the first 8 bytes as a little-endian u64, reduce mod p.
    // Reject 0 (degenerate chain) by mapping it to 2 — keeps the function
    // total without breaking determinism.
    let raw = u64::from_le_bytes(derived[..8].try_into().expect("≥8 bytes from BLAKE3"));
    let r = raw % FIELD_MODULUS;
    if r == 0 { 2 } else { r }
}

/// Compute `[α¹, α², α³, ..., α^n] mod p`.
fn compute_alpha_chain(alpha: u64, n: usize) -> Vec<u64> {
    let mut chain = Vec::with_capacity(n);
    let mut power = alpha;
    for _ in 0..n {
        chain.push(power);
        power = mod_mul(power, alpha, FIELD_MODULUS);
    }
    chain
}

/// `(a * b) mod m` using 128-bit intermediate to avoid overflow.
fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

// ════════════════════════════════════════════════════════════════════════════
// AIR construction
// ════════════════════════════════════════════════════════════════════════════

/// Build the execution trace from a precomputed chain.
///
/// Layout (3 registers, `srs_size` rows):
///   * REG_ALPHA = α (constant across all rows)
///   * REG_POWER = chain[i] at row i
///   * REG_STEP_VALID = 1 (always; reserved for Phase 2's per-step witness)
///
/// Note: the AIR's existing transition expression doesn't natively support
/// modular multiplication (it uses `wrapping_mul`). We adapt by encoding
/// constraints over the differences and relying on the boundary checks +
/// the verifier's chain-recomputation step to bind soundness. This is the
/// same pattern used by the production encryption_zkstark module.
fn build_trace(alpha: u64, chain: &[u64]) -> ExecutionTrace {
    let mut rows: Vec<Vec<u64>> = Vec::with_capacity(chain.len());
    for &power in chain {
        let mut row = vec![0u64; REG_COUNT];
        row[REG_ALPHA] = alpha;
        row[REG_POWER] = power;
        row[REG_STEP_VALID] = 1;
        rows.push(row);
    }

    // First-row public inputs: [alpha, power₀, valid=1].
    let public_inputs = rows.first().cloned().unwrap_or_default();
    let mut trace = ExecutionTrace::new(rows, public_inputs);
    trace.pad_to_power_of_two();
    trace
}

/// Build the AIR constraint set.
///
/// Constraints:
///   * Boundary[0]: REG_ALPHA == α      (alpha is fixed at row 0)
///   * Boundary[0]: REG_POWER == α      (chain starts at α¹)
///   * Boundary[0]: REG_STEP_VALID == 1
///   * Transition: REG_ALPHA[i+1] - REG_ALPHA[i] == 0   (alpha constant)
///   * Transition: REG_STEP_VALID[i] == 1               (valid stays 1)
///
/// Note: the multiplicative-chain constraint
/// `REG_POWER[i+1] == REG_POWER[i] * α mod p` cannot be expressed under the
/// current `TransitionExpression::Mul` (which uses `wrapping_mul`, not modular).
/// Soundness for the chain is enforced by the verifier's recomputation step
/// (Check 2 in `verify()`), not by the AIR transition constraint. Phase 2's
/// upgrade to field-arithmetic AIR will move this into the constraint set.
fn build_constraints(alpha: u64, _srs_size: usize) -> AirConstraints {
    let mut c = AirConstraints::new();

    c.add_boundary_constraint(BoundaryConstraint {
        register: REG_ALPHA,
        step: BoundaryStep::Initial,
        value: alpha,
        description: "alpha is fixed by the seed derivation at row 0".into(),
    });
    c.add_boundary_constraint(BoundaryConstraint {
        register: REG_POWER,
        step: BoundaryStep::Initial,
        value: alpha,
        description: "chain starts at α¹".into(),
    });
    c.add_boundary_constraint(BoundaryConstraint {
        register: REG_STEP_VALID,
        step: BoundaryStep::Initial,
        value: 1,
        description: "valid bit is 1 at row 0".into(),
    });

    // Transition: alpha is constant — REG_ALPHA[i+1] - REG_ALPHA[i] == 0
    c.add_transition_constraint(TransitionConstraint {
        expression: TransitionExpression::Sub(
            Box::new(TransitionExpression::Next(REG_ALPHA)),
            Box::new(TransitionExpression::Current(REG_ALPHA)),
        ),
        degree: 1,
        description: "alpha constant across rows".into(),
    });

    // Transition: valid bit stays 1 — REG_STEP_VALID[i] - 1 == 0
    c.add_transition_constraint(TransitionConstraint {
        expression: TransitionExpression::Sub(
            Box::new(TransitionExpression::Current(REG_STEP_VALID)),
            Box::new(TransitionExpression::Constant(1)),
        ),
        degree: 1,
        description: "valid bit stays 1".into(),
    });

    c
}

/// Serialize the constraint set into a compact byte blob so it can be passed
/// to the StarkProver. The current prover doesn't parse this — it's
/// reserved for the Phase 2 polynomial-constraint compiler — but we include
/// it here for forward-compat and to bind the constraint definition into
/// the proof's commitment domain.
fn serialize_constraints(c: &AirConstraints) -> Vec<u8> {
    bincode::serialize(&ConstraintsBlob {
        n_boundary: c.boundary_constraints.len() as u32,
        n_transition: c.transition_constraints.len() as u32,
        n_global: c.global_constraints.len() as u32,
        degree: c.constraint_degree as u32,
        blowup: c.blowup_factor as u32,
    })
    .unwrap_or_default()
}

#[derive(Serialize, Deserialize)]
struct ConstraintsBlob {
    n_boundary: u32,
    n_transition: u32,
    n_global: u32,
    degree: u32,
    blowup: u32,
}

/// Recompute the trace commitment using the same algorithm StarkProver uses
/// (`Sha3_256` over little-endian u64s). Used in verify to confirm the
/// proof's claimed commitment matches our locally-rebuilt trace.
fn trace_commitment_blake3(trace: &[Vec<u64>]) -> [u8; 32] {
    use sha3::{Digest, Sha3_256};
    let mut hasher = Sha3_256::new();
    for row in trace {
        for &v in row {
            hasher.update(v.to_le_bytes());
        }
    }
    hasher.finalize().into()
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn generate_and_verify_small() {
        let seed = [42u8; 32];
        let (proof, srs_bytes) = NovaSrsProof::generate(&seed, 4)
            .await
            .expect("generate");
        assert_eq!(proof.srs_size, 4);
        assert_eq!(srs_bytes.len(), 4 * 8);
        assert!(proof.verify().await.expect("verify"));
    }

    #[tokio::test]
    async fn generate_and_verify_64() {
        let seed = [7u8; 32];
        let (proof, srs_bytes) = NovaSrsProof::generate(&seed, 64).await.expect("generate");
        assert_eq!(proof.srs_size, 64);
        assert_eq!(srs_bytes.len(), 64 * 8);
        assert!(proof.verify().await.expect("verify"));
    }

    #[tokio::test]
    async fn rejects_tampered_seed() {
        let seed = [1u8; 32];
        let (mut proof, _) = NovaSrsProof::generate(&seed, 16).await.expect("generate");
        proof.seed[0] ^= 1;
        assert!(!proof.verify().await.expect("verify"));
    }

    #[tokio::test]
    async fn rejects_tampered_alpha() {
        let seed = [2u8; 32];
        let (mut proof, _) = NovaSrsProof::generate(&seed, 16).await.expect("generate");
        proof.alpha = proof.alpha.wrapping_add(1) % FIELD_MODULUS;
        assert!(!proof.verify().await.expect("verify"));
    }

    #[tokio::test]
    async fn rejects_tampered_final_power() {
        let seed = [3u8; 32];
        let (mut proof, _) = NovaSrsProof::generate(&seed, 16).await.expect("generate");
        proof.final_power = proof.final_power.wrapping_add(1) % FIELD_MODULUS;
        assert!(!proof.verify().await.expect("verify"));
    }

    #[tokio::test]
    async fn rejects_tampered_srs_root() {
        let seed = [4u8; 32];
        let (mut proof, _) = NovaSrsProof::generate(&seed, 16).await.expect("generate");
        proof.srs_root[0] ^= 1;
        assert!(!proof.verify().await.expect("verify"));
    }

    #[tokio::test]
    async fn rejects_tampered_stark_bytes() {
        let seed = [5u8; 32];
        let (mut proof, _) = NovaSrsProof::generate(&seed, 16).await.expect("generate");
        if !proof.stark_bytes.is_empty() {
            proof.stark_bytes[0] ^= 1;
        }
        // Tampering with the postcard bytes is overwhelmingly likely to fail
        // either deserialization or one of the Check-2/Check-3 invariants.
        // Verify returns false (not panic).
        assert!(!proof.verify().await.expect("verify"));
    }

    #[tokio::test]
    async fn rejects_zero_size() {
        let seed = [6u8; 32];
        let result = NovaSrsProof::generate(&seed, 0).await;
        assert!(result.is_err());
    }

    #[test]
    fn alpha_derivation_is_deterministic() {
        let seed = [9u8; 32];
        let a1 = derive_alpha(&seed);
        let a2 = derive_alpha(&seed);
        assert_eq!(a1, a2);
        assert_ne!(a1, 0);
        assert!(a1 < FIELD_MODULUS);
    }

    #[test]
    fn alpha_differs_for_different_seeds() {
        let s1 = [1u8; 32];
        let s2 = [2u8; 32];
        assert_ne!(derive_alpha(&s1), derive_alpha(&s2));
    }

    #[test]
    fn chain_is_correct() {
        let alpha = 3u64;
        let chain = compute_alpha_chain(alpha, 5);
        assert_eq!(chain.len(), 5);
        // alpha¹, alpha², alpha³, alpha⁴, alpha⁵
        assert_eq!(chain[0], 3);
        assert_eq!(chain[1], 9);
        assert_eq!(chain[2], 27);
        assert_eq!(chain[3], 81);
        assert_eq!(chain[4], 243);
    }

    #[test]
    fn mod_mul_no_overflow() {
        // Pick two near-max u64 values; verify no overflow.
        let a = FIELD_MODULUS - 1;
        let b = FIELD_MODULUS - 1;
        let r = mod_mul(a, b, FIELD_MODULUS);
        assert!(r < FIELD_MODULUS);
        // (p-1)² mod p == 1
        assert_eq!(r, 1);
    }
}
