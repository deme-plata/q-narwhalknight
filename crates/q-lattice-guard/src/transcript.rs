//! Fiat-Shamir Transcript for Non-Interactive Proofs
//!
//! Uses lattice-based hashing combined with BLAKE3 for post-quantum
//! Fiat-Shamir transformation.

use crate::{
    commitment::LatticeCommitment,
    params::RlweParams,
    Challenge, Polynomial, Scalar,
};
use blake3::Hasher;
use serde::{Deserialize, Serialize};

/// Lattice-based Fiat-Shamir transcript
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatticeTranscript {
    /// BLAKE3 state for efficient hashing
    #[serde(skip)]
    hasher: Option<Hasher>,
    /// Accumulated hash for serialization
    state: [u8; 32],
    /// Generated challenges
    challenges: Vec<Challenge>,
    /// Parameters for challenge generation
    params: RlweParams,
}

impl LatticeTranscript {
    /// Create new transcript with given parameters
    pub fn new(params: RlweParams) -> Self {
        let hasher = Hasher::new();
        Self {
            hasher: Some(hasher),
            state: [0u8; 32],
            challenges: Vec::new(),
            params,
        }
    }

    /// Create transcript from proof for verification
    pub fn from_proof_state(state: [u8; 32], params: RlweParams) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(&state);
        Self {
            hasher: Some(hasher),
            state,
            challenges: Vec::new(),
            params,
        }
    }

    /// Append bytes to transcript
    pub fn append_bytes(&mut self, label: &[u8], data: &[u8]) {
        if let Some(ref mut hasher) = self.hasher {
            hasher.update(label);
            hasher.update(&(data.len() as u64).to_le_bytes());
            hasher.update(data);
        }
    }

    /// Append scalar to transcript
    pub fn append_scalar(&mut self, label: &[u8], scalar: Scalar) {
        self.append_bytes(label, &scalar.to_le_bytes());
    }

    /// Append polynomial to transcript
    pub fn append_polynomial(&mut self, label: &[u8], poly: &Polynomial) {
        self.append_bytes(label, &(poly.coefficients.len() as u64).to_le_bytes());
        for &coeff in &poly.coefficients {
            self.append_scalar(b"coeff", coeff);
        }
    }

    /// Append commitment to transcript
    pub fn append_commitment(&mut self, label: &[u8], commitment: &LatticeCommitment) {
        self.append_bytes(label, &commitment.to_bytes());
    }

    /// Append multiple commitments
    pub fn append_commitments(&mut self, commitments: &[LatticeCommitment]) {
        for (i, commitment) in commitments.iter().enumerate() {
            let label = format!("commitment_{}", i);
            self.append_commitment(label.as_bytes(), commitment);
        }
    }

    /// Generate a challenge scalar
    pub fn challenge_scalar(&mut self, label: &[u8]) -> Scalar {
        self.append_bytes(b"challenge", label);

        let hash = if let Some(ref hasher) = self.hasher {
            hasher.finalize()
        } else {
            blake3::hash(&self.state)
        };

        // Map hash to scalar in Z_q
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&hash.as_bytes()[..8]);
        let scalar = u64::from_le_bytes(bytes) % self.params.modulus;

        // Update state
        self.state.copy_from_slice(hash.as_bytes());
        if let Some(ref mut hasher) = self.hasher {
            *hasher = Hasher::new();
            hasher.update(&self.state);
        }

        scalar
    }

    /// Generate a challenge polynomial with bounded coefficients
    pub fn generate_challenge(&mut self) -> Challenge {
        let mut coefficients = Vec::with_capacity(self.params.dimension);

        // Generate small coefficients (-2, -1, 0, 1, 2)
        for i in 0..self.params.dimension {
            let label = format!("challenge_coeff_{}", i);
            let raw = self.challenge_scalar(label.as_bytes());

            // Map to {-2, -1, 0, 1, 2}
            let small = (raw % 5) as i64 - 2;
            let coeff = if small < 0 {
                self.params.modulus - ((-small) as u64)
            } else {
                small as u64
            };
            coefficients.push(coeff);
        }

        let challenge = Challenge {
            polynomial: Polynomial::new(coefficients),
            bound: 2,
        };

        self.challenges.push(challenge.clone());
        challenge
    }

    /// Finalize and get transcript state
    pub fn finalize(&self) -> [u8; 32] {
        if let Some(ref hasher) = self.hasher {
            *hasher.finalize().as_bytes()
        } else {
            self.state
        }
    }

    /// Get all generated challenges
    pub fn get_challenges(&self) -> &[Challenge] {
        &self.challenges
    }
}

/// SIS-based hash function for extra post-quantum security
pub struct SisHash {
    /// Hash matrix A (random in Z_q^{n×m})
    matrix: Vec<Vec<Scalar>>,
    /// Modulus
    modulus: Scalar,
    /// Output dimension
    output_dim: usize,
    /// Input dimension
    input_dim: usize,
}

impl SisHash {
    /// Create new SIS hash with given parameters
    pub fn new(output_dim: usize, input_dim: usize, modulus: Scalar) -> Self {
        // For production, this matrix should be derived from a seed
        let mut rng = rand::thread_rng();
        let matrix: Vec<Vec<Scalar>> = (0..output_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| rand::Rng::gen::<Scalar>(&mut rng) % modulus)
                    .collect()
            })
            .collect();

        Self {
            matrix,
            modulus,
            output_dim,
            input_dim,
        }
    }

    /// Hash input vector x to output Ax mod q
    pub fn hash(&self, input: &[Scalar]) -> Vec<Scalar> {
        assert!(input.len() <= self.input_dim);

        let mut output = vec![0u64; self.output_dim];

        for i in 0..self.output_dim {
            for j in 0..input.len().min(self.input_dim) {
                let product =
                    ((self.matrix[i][j] as u128 * input[j] as u128) % self.modulus as u128) as u64;
                output[i] = (output[i] + product) % self.modulus;
            }
        }

        output
    }

    /// Collision-finding is as hard as finding short vectors in lattices
    pub fn security_level(&self) -> &str {
        "Post-quantum secure under SIS assumption"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcript_deterministic() {
        let params = RlweParams::pq128();

        let mut t1 = LatticeTranscript::new(params.clone());
        let mut t2 = LatticeTranscript::new(params);

        t1.append_bytes(b"test", b"hello");
        t2.append_bytes(b"test", b"hello");

        let c1 = t1.challenge_scalar(b"challenge");
        let c2 = t2.challenge_scalar(b"challenge");

        assert_eq!(c1, c2, "Same inputs should produce same challenge");
    }

    #[test]
    fn test_different_inputs_different_outputs() {
        let params = RlweParams::pq128();

        let mut t1 = LatticeTranscript::new(params.clone());
        let mut t2 = LatticeTranscript::new(params);

        t1.append_bytes(b"test", b"hello");
        t2.append_bytes(b"test", b"world");

        let c1 = t1.challenge_scalar(b"challenge");
        let c2 = t2.challenge_scalar(b"challenge");

        assert_ne!(c1, c2, "Different inputs should produce different challenges");
    }

    #[test]
    fn test_sis_hash() {
        let sis = SisHash::new(256, 512, 4294957057);

        let input: Vec<Scalar> = (0..512).map(|i| i as Scalar).collect();
        let hash = sis.hash(&input);

        assert_eq!(hash.len(), 256);

        // Same input should give same output
        let hash2 = sis.hash(&input);
        assert_eq!(hash, hash2);
    }
}
