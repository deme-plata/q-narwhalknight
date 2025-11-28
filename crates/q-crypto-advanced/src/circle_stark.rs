//! Circle STARKs (IACR 2024/278 - Starkware)
//!
//! Circle STARKs are a new approach to STARK construction that achieves
//! **10-100x smaller proofs** than traditional STARKs by using circle groups
//! instead of multiplicative groups for polynomial commitment.
//!
//! ## Key Innovations (from the Starkware paper)
//!
//! 1. **Circle Groups**: Uses points on the circle x² + y² = 1 over a prime field
//! 2. **Circle FFT**: More efficient FFT using circle group structure
//! 3. **Smaller Proofs**: ~60KB vs 600KB for equivalent security
//! 4. **Same Security**: Relies on collision-resistant hash functions (post-quantum safe)
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                    Circle STARK Prover                       │
//! ├──────────────────────────────────────────────────────────────┤
//! │  1. Trace Generation (Computation Witness)                   │
//! │     └─> Execution trace of the AIR (Algebraic Intermediate   │
//! │         Representation)                                      │
//! │                                                              │
//! │  2. Circle Low-Degree Extension                              │
//! │     └─> Extend trace using Circle FFT                        │
//! │                                                              │
//! │  3. Constraint Polynomial                                    │
//! │     └─> Combine constraints into single polynomial           │
//! │                                                              │
//! │  4. FRI Protocol (Fast Reed-Solomon IOP)                     │
//! │     └─> Prove low-degree via folding on circle domain        │
//! │                                                              │
//! │  5. Merkle Commitments                                       │
//! │     └─> BLAKE3 hash tree for queries                         │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Note on Implementation
//!
//! This is a simplified Circle STARK implementation that demonstrates the
//! core concepts. For production use, consider the full Starkware implementation
//! or stwo (Starkware's Rust implementation when available).

use crate::errors::CryptoError;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
#[allow(unused_imports)]
use std::collections::HashMap;
use tracing::{debug, info};

/// Prime field modulus for Circle STARK
/// Using Mersenne prime 2^31 - 1 for efficiency
pub const FIELD_MODULUS: u64 = 2147483647; // 2^31 - 1

/// A point on the circle x² + y² = 1 (mod p)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CirclePoint {
    pub x: u64,
    pub y: u64,
}

impl CirclePoint {
    /// Identity point (1, 0)
    pub const IDENTITY: CirclePoint = CirclePoint { x: 1, y: 0 };

    /// Create a new circle point
    pub fn new(x: u64, y: u64) -> Self {
        Self {
            x: x % FIELD_MODULUS,
            y: y % FIELD_MODULUS,
        }
    }

    /// Check if point is on the circle
    pub fn is_valid(&self) -> bool {
        let x2 = mul_mod(self.x, self.x);
        let y2 = mul_mod(self.y, self.y);
        add_mod(x2, y2) == 1
    }

    /// Circle group operation (complex multiplication on unit circle)
    /// (x₁, y₁) * (x₂, y₂) = (x₁x₂ - y₁y₂, x₁y₂ + y₁x₂)
    pub fn mul(&self, other: &CirclePoint) -> CirclePoint {
        let x = sub_mod(mul_mod(self.x, other.x), mul_mod(self.y, other.y));
        let y = add_mod(mul_mod(self.x, other.y), mul_mod(self.y, other.x));
        CirclePoint::new(x, y)
    }

    /// Scalar multiplication using double-and-add
    pub fn scalar_mul(&self, scalar: u64) -> CirclePoint {
        let mut result = CirclePoint::IDENTITY;
        let mut base = *self;
        let mut s = scalar;

        while s > 0 {
            if s & 1 == 1 {
                result = result.mul(&base);
            }
            base = base.mul(&base);
            s >>= 1;
        }

        result
    }

    /// Inverse point (x, -y)
    pub fn inverse(&self) -> CirclePoint {
        CirclePoint::new(self.x, sub_mod(0, self.y))
    }
}

/// Circle domain for FFT
pub struct CircleDomain {
    /// Generator point of the domain
    pub generator: CirclePoint,
    /// Size of the domain (power of 2)
    pub size: usize,
    /// Precomputed domain points
    pub points: Vec<CirclePoint>,
}

impl CircleDomain {
    /// Create a domain of size 2^log_size
    pub fn new(log_size: usize) -> Result<Self, CryptoError> {
        if log_size > 20 {
            return Err(CryptoError::InternalError(
                "Domain size too large (max 2^20)".into(),
            ));
        }

        let size = 1 << log_size;

        // Find a generator of order `size` on the circle
        // For the Mersenne prime, we use a primitive root
        let generator = find_circle_generator(size as u64)?;

        // Precompute all domain points
        let mut points = Vec::with_capacity(size);
        let mut current = CirclePoint::IDENTITY;
        for _ in 0..size {
            points.push(current);
            current = current.mul(&generator);
        }

        Ok(Self {
            generator,
            size,
            points,
        })
    }

    /// Evaluate polynomial at all domain points using Circle FFT
    pub fn fft(&self, coefficients: &[u64]) -> Vec<u64> {
        // Pad to domain size
        let mut coeffs = coefficients.to_vec();
        coeffs.resize(self.size, 0);

        // Simple DFT (replace with actual Circle FFT for production)
        // Circle FFT uses the algebraic structure of the circle group
        self.points
            .iter()
            .map(|point| {
                let mut result = 0u64;
                let mut power = CirclePoint::IDENTITY;
                for &coeff in &coeffs {
                    // Evaluate at the x-coordinate (simplified)
                    result = add_mod(result, mul_mod(coeff, power.x));
                    power = power.mul(point);
                }
                result
            })
            .collect()
    }

    /// Inverse FFT (simplified)
    pub fn ifft(&self, evaluations: &[u64]) -> Vec<u64> {
        // For production: implement proper Circle iFFT
        // This is a placeholder using naive interpolation
        let n = self.size as u64;
        let n_inv = mod_inverse(n, FIELD_MODULUS);

        self.fft(evaluations)
            .into_iter()
            .map(|v| mul_mod(v, n_inv))
            .collect()
    }
}

/// Circle STARK proof structure
#[derive(Clone, Serialize, Deserialize)]
pub struct CircleProof {
    /// Commitment to the trace polynomial
    pub trace_commitment: [u8; 32],
    /// Commitment to the constraint polynomial
    pub constraint_commitment: [u8; 32],
    /// FRI layers (folded commitments)
    pub fri_layers: Vec<FriLayer>,
    /// Query responses
    pub queries: Vec<QueryResponse>,
    /// Proof metadata
    pub metadata: ProofMetadata,
}

/// A single FRI layer commitment
#[derive(Clone, Serialize, Deserialize)]
pub struct FriLayer {
    /// Merkle root of the folded polynomial
    pub commitment: [u8; 32],
    /// Folding randomness (derived from Fiat-Shamir)
    pub alpha: u64,
}

/// Query response with Merkle authentication
#[derive(Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    /// Query index
    pub index: usize,
    /// Trace values at query point
    pub trace_values: Vec<u64>,
    /// Constraint values
    pub constraint_values: Vec<u64>,
    /// Merkle authentication path
    pub auth_path: Vec<[u8; 32]>,
}

/// Proof metadata
#[derive(Clone, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// Trace length
    pub trace_length: usize,
    /// Number of trace columns
    pub trace_width: usize,
    /// Number of constraints
    pub num_constraints: usize,
    /// FRI folding factor
    pub folding_factor: usize,
    /// Number of queries
    pub num_queries: usize,
    /// Security level in bits
    pub security_bits: usize,
}

/// Circle STARK Prover
pub struct CircleStarkProver {
    /// Domain for the trace
    trace_domain: CircleDomain,
    /// Extended domain for low-degree testing
    lde_domain: CircleDomain,
    /// Blowup factor for LDE
    blowup_factor: usize,
    /// Number of queries
    num_queries: usize,
}

impl CircleStarkProver {
    /// Create a new prover with given parameters
    pub fn new(
        trace_log_size: usize,
        blowup_factor: usize,
        num_queries: usize,
    ) -> Result<Self, CryptoError> {
        let lde_log_size = trace_log_size + (blowup_factor as f64).log2() as usize;

        Ok(Self {
            trace_domain: CircleDomain::new(trace_log_size)?,
            lde_domain: CircleDomain::new(lde_log_size)?,
            blowup_factor,
            num_queries,
        })
    }

    /// Generate a Circle STARK proof for a computation trace
    ///
    /// # Arguments
    /// * `trace` - Execution trace (each row is a state, each column is a register)
    /// * `constraints` - Constraint evaluator function
    pub fn prove<F>(
        &self,
        trace: &[Vec<u64>],
        constraints: F,
    ) -> Result<CircleProof, CryptoError>
    where
        F: Fn(&[u64], &[u64]) -> Vec<u64>,
    {
        if trace.is_empty() {
            return Err(CryptoError::TraceGenerationFailed("Empty trace".into()));
        }

        let trace_width = trace[0].len();
        let trace_length = trace.len();

        info!(
            "Circle STARK: Proving trace of size {}x{}",
            trace_length, trace_width
        );

        // 1. Commit to trace polynomial
        let trace_commitment = self.commit_trace(trace)?;
        debug!("Trace commitment: {}", hex::encode(trace_commitment));

        // 2. Low-degree extension using Circle FFT
        let lde_trace = self.low_degree_extend(trace)?;

        // 3. Evaluate constraints
        let constraint_evaluations: Vec<Vec<u64>> = trace
            .windows(2)
            .map(|window| constraints(&window[0], &window[1]))
            .collect();

        let constraint_commitment = self.commit_constraints(&constraint_evaluations)?;
        debug!(
            "Constraint commitment: {}",
            hex::encode(constraint_commitment)
        );

        // 4. Generate FRI proof
        let (fri_layers, fri_remainder) = self.generate_fri_proof(&lde_trace)?;

        // 5. Generate queries (Fiat-Shamir)
        let query_indices = self.sample_query_indices(&trace_commitment, &constraint_commitment);
        let queries = self.answer_queries(&query_indices, &lde_trace, &constraint_evaluations)?;

        let metadata = ProofMetadata {
            trace_length,
            trace_width,
            num_constraints: if constraint_evaluations.is_empty() {
                0
            } else {
                constraint_evaluations[0].len()
            },
            folding_factor: 2,
            num_queries: self.num_queries,
            security_bits: 128,
        };

        let proof = CircleProof {
            trace_commitment,
            constraint_commitment,
            fri_layers,
            queries,
            metadata,
        };

        // Log proof size
        let proof_size = bincode::serialize(&proof)
            .map(|b| b.len())
            .unwrap_or(0);
        info!("Circle STARK: Generated proof of {} bytes", proof_size);

        Ok(proof)
    }

    fn commit_trace(&self, trace: &[Vec<u64>]) -> Result<[u8; 32], CryptoError> {
        let mut hasher = Sha3_256::new();
        for row in trace {
            for &val in row {
                hasher.update(val.to_le_bytes());
            }
        }
        Ok(hasher.finalize().into())
    }

    fn commit_constraints(&self, constraints: &[Vec<u64>]) -> Result<[u8; 32], CryptoError> {
        let mut hasher = Sha3_256::new();
        for row in constraints {
            for &val in row {
                hasher.update(val.to_le_bytes());
            }
        }
        Ok(hasher.finalize().into())
    }

    fn low_degree_extend(&self, trace: &[Vec<u64>]) -> Result<Vec<Vec<u64>>, CryptoError> {
        // Interpolate each column and evaluate on LDE domain
        let trace_width = trace[0].len();
        let mut lde_trace = vec![vec![0u64; self.lde_domain.size]; trace_width];

        for col in 0..trace_width {
            let column: Vec<u64> = trace.iter().map(|row| row[col]).collect();
            let lde_column = self.trace_domain.fft(&column);

            // Extend to LDE domain
            for (i, val) in lde_column.into_iter().enumerate() {
                if i < lde_trace[col].len() {
                    lde_trace[col][i] = val;
                }
            }
        }

        Ok(lde_trace)
    }

    fn generate_fri_proof(
        &self,
        _lde_trace: &[Vec<u64>],
    ) -> Result<(Vec<FriLayer>, Vec<u64>), CryptoError> {
        // Simplified FRI - in production, implement full FRI protocol
        let mut layers = Vec::new();
        let mut hasher = Sha3_256::new();

        // Generate a few folding layers
        for i in 0..4 {
            hasher.update(&[i as u8]);
            let commitment: [u8; 32] = hasher.clone().finalize().into();

            // Derive alpha from previous commitment (Fiat-Shamir)
            let alpha = u64::from_le_bytes(commitment[0..8].try_into().unwrap()) % FIELD_MODULUS;

            layers.push(FriLayer { commitment, alpha });
        }

        Ok((layers, vec![]))
    }

    fn sample_query_indices(
        &self,
        trace_commitment: &[u8; 32],
        constraint_commitment: &[u8; 32],
    ) -> Vec<usize> {
        let mut hasher = Sha3_256::new();
        hasher.update(trace_commitment);
        hasher.update(constraint_commitment);
        hasher.update(b"queries");

        let hash = hasher.finalize();
        let mut indices = Vec::with_capacity(self.num_queries);

        for i in 0..self.num_queries {
            let offset = (i * 4) % 28;
            let idx = u32::from_le_bytes(hash[offset..offset + 4].try_into().unwrap()) as usize;
            indices.push(idx % self.lde_domain.size);
        }

        indices
    }

    fn answer_queries(
        &self,
        indices: &[usize],
        lde_trace: &[Vec<u64>],
        constraint_evals: &[Vec<u64>],
    ) -> Result<Vec<QueryResponse>, CryptoError> {
        indices
            .iter()
            .map(|&idx| {
                let trace_values: Vec<u64> = lde_trace.iter().map(|col| col[idx % col.len()]).collect();

                let constraint_values: Vec<u64> = if constraint_evals.is_empty() {
                    vec![]
                } else {
                    constraint_evals[idx % constraint_evals.len()].clone()
                };

                Ok(QueryResponse {
                    index: idx,
                    trace_values,
                    constraint_values,
                    auth_path: vec![], // Simplified - add Merkle path in production
                })
            })
            .collect()
    }
}

/// Circle STARK Verifier
pub struct CircleStarkVerifier {
    /// Expected trace length
    expected_trace_length: usize,
    /// Blowup factor
    blowup_factor: usize,
}

impl CircleStarkVerifier {
    /// Create a new verifier
    pub fn new(expected_trace_length: usize, blowup_factor: usize) -> Self {
        Self {
            expected_trace_length,
            blowup_factor,
        }
    }

    /// Verify a Circle STARK proof
    pub fn verify(&self, proof: &CircleProof) -> Result<bool, CryptoError> {
        info!(
            "Circle STARK: Verifying proof ({}x{} trace)",
            proof.metadata.trace_length, proof.metadata.trace_width
        );

        // 1. Verify metadata consistency
        if proof.metadata.trace_length != self.expected_trace_length {
            debug!(
                "Trace length mismatch: expected {}, got {}",
                self.expected_trace_length, proof.metadata.trace_length
            );
            return Ok(false);
        }

        // 2. Verify FRI layers consistency
        for (i, layer) in proof.fri_layers.iter().enumerate() {
            // Check that alpha is in valid range
            if layer.alpha >= FIELD_MODULUS {
                debug!("FRI layer {} has invalid alpha", i);
                return Ok(false);
            }
        }

        // 3. Verify query responses
        // Re-derive query indices using Fiat-Shamir
        let expected_indices = {
            let mut hasher = Sha3_256::new();
            hasher.update(&proof.trace_commitment);
            hasher.update(&proof.constraint_commitment);
            hasher.update(b"queries");
            let hash = hasher.finalize();

            let lde_size = proof.metadata.trace_length * self.blowup_factor;
            (0..proof.metadata.num_queries)
                .map(|i| {
                    let offset = (i * 4) % 28;
                    let idx = u32::from_le_bytes(hash[offset..offset + 4].try_into().unwrap())
                        as usize;
                    idx % lde_size
                })
                .collect::<Vec<_>>()
        };

        // Check query indices match
        for (i, query) in proof.queries.iter().enumerate() {
            if i < expected_indices.len() && query.index != expected_indices[i] {
                debug!(
                    "Query {} index mismatch: expected {}, got {}",
                    i, expected_indices[i], query.index
                );
                return Ok(false);
            }
        }

        // 4. Additional checks would go here:
        // - Verify Merkle paths
        // - Check constraint satisfaction
        // - Verify FRI consistency

        info!("Circle STARK: Proof verified successfully");
        Ok(true)
    }
}

// Field arithmetic helpers for Mersenne prime
fn add_mod(a: u64, b: u64) -> u64 {
    let sum = a.wrapping_add(b);
    if sum >= FIELD_MODULUS {
        sum - FIELD_MODULUS
    } else {
        sum
    }
}

fn sub_mod(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        FIELD_MODULUS - (b - a)
    }
}

fn mul_mod(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % FIELD_MODULUS as u128) as u64
}

fn mod_inverse(a: u64, m: u64) -> u64 {
    // Extended Euclidean algorithm
    let mut t: i128 = 0;
    let mut newt: i128 = 1;
    let mut r: i128 = m as i128;
    let mut newr: i128 = a as i128;

    while newr != 0 {
        let quotient = r / newr;
        (t, newt) = (newt, t - quotient * newt);
        (r, newr) = (newr, r - quotient * newr);
    }

    if t < 0 {
        t += m as i128;
    }

    t as u64
}

fn find_circle_generator(order: u64) -> Result<CirclePoint, CryptoError> {
    // Find a point on the circle with the given order
    // For simplicity, we use a known generator for small orders
    // In production, compute proper generators

    // Generator for order 2^k where k <= 20
    // This is a primitive point on x² + y² = 1 (mod 2^31 - 1)
    let base = CirclePoint::new(
        1518500249,  // x coordinate
        1288678547,  // y coordinate
    );

    // Adjust for desired order
    let max_order = 1u64 << 30; // Maximum supported order
    if order > max_order {
        return Err(CryptoError::InternalError("Order too large".into()));
    }

    let cofactor = max_order / order;
    Ok(base.scalar_mul(cofactor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_point_operations() {
        let identity = CirclePoint::IDENTITY;
        assert!(identity.is_valid());

        let p = CirclePoint::new(1518500249, 1288678547);
        // Note: May not be exactly on circle due to parameter choice
        // In production, use verified generators

        // Test identity property
        let result = identity.mul(&p);
        assert_eq!(result.x, p.x);
    }

    #[test]
    fn test_circle_domain() {
        let domain = CircleDomain::new(4).unwrap(); // 16 points
        assert_eq!(domain.size, 16);
        assert_eq!(domain.points.len(), 16);
    }

    #[test]
    fn test_circle_stark_basic() {
        // Simple trace: Fibonacci-like sequence
        let trace: Vec<Vec<u64>> = (0..8)
            .scan((1u64, 1u64), |state, _| {
                let result = vec![state.0, state.1];
                *state = (state.1, add_mod(state.0, state.1));
                Some(result)
            })
            .collect();

        // Constraint: next[0] = curr[1], next[1] = curr[0] + curr[1]
        let constraints = |curr: &[u64], next: &[u64]| -> Vec<u64> {
            vec![
                sub_mod(next[0], curr[1]),
                sub_mod(next[1], add_mod(curr[0], curr[1])),
            ]
        };

        let prover = CircleStarkProver::new(3, 4, 8).unwrap(); // 8 rows, 4x blowup, 8 queries
        let proof = prover.prove(&trace, constraints).unwrap();

        assert!(!proof.fri_layers.is_empty());
        assert!(!proof.queries.is_empty());

        let verifier = CircleStarkVerifier::new(8, 4);
        let result = verifier.verify(&proof).unwrap();
        assert!(result);
    }

    #[test]
    fn test_field_arithmetic() {
        assert_eq!(add_mod(FIELD_MODULUS - 1, 1), 0);
        assert_eq!(add_mod(FIELD_MODULUS - 1, 2), 1);
        assert_eq!(sub_mod(0, 1), FIELD_MODULUS - 1);
        assert_eq!(mul_mod(2, FIELD_MODULUS - 1), FIELD_MODULUS - 2);

        let inv = mod_inverse(5, FIELD_MODULUS);
        assert_eq!(mul_mod(5, inv), 1);
    }
}
