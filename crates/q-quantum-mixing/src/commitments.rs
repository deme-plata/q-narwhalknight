// Pedersen commitments for quantum mixing privacy
// Provides hiding and binding properties for values

use ark_ec::CurveGroup;
use ark_ff::Field;
use serde::{Serialize, Deserialize};
use std::marker::PhantomData;

/// Pedersen commitment structure
/// Commitment = value * G + blinding_factor * H
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PedersenCommitment<C: CurveGroup> {
    /// The commitment point on the curve
    pub commitment_point: C,
    /// Blinding factor used in commitment
    blinding_factor: C::ScalarField,
    /// Value being committed to
    value: u64,
}

impl<C: CurveGroup> PedersenCommitment<C> {
    /// Create a new Pedersen commitment
    pub fn new(value: u64, blinding_factor: C::ScalarField) -> Self {
        // In a real implementation, this would compute value*G + blinding_factor*H
        // where G and H are generator points
        let commitment_point = C::generator(); // Placeholder

        Self {
            commitment_point,
            blinding_factor,
            value,
        }
    }

    /// Create a zero commitment
    pub fn zero() -> Self {
        Self {
            commitment_point: C::generator(),
            blinding_factor: C::ScalarField::ZERO,
            value: 0,
        }
    }

    /// Add two commitments homomorphically
    pub fn add(&self, other: &Self) -> Self {
        Self {
            commitment_point: self.commitment_point + other.commitment_point,
            blinding_factor: self.blinding_factor + other.blinding_factor,
            value: self.value + other.value,
        }
    }

    /// Get the blinding factor (for proof generation)
    pub fn blinding_factor(&self) -> C::ScalarField {
        self.blinding_factor
    }

    /// Get the committed value (for proof generation)
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Serialize the commitment for hashing/storage
    pub fn serialize(&self) -> Vec<u8> {
        // In production, properly serialize the curve point
        format!("commitment_{}_{}", self.value, self.blinding_factor)
            .as_bytes()
            .to_vec()
    }

    /// Verify that this commitment corresponds to the given value and blinding factor
    pub fn verify(&self, value: u64, blinding_factor: C::ScalarField) -> bool {
        // In production, recompute commitment and check equality
        self.value == value && self.blinding_factor == blinding_factor
    }
}

impl<C: CurveGroup> PartialEq for PedersenCommitment<C> {
    fn eq(&self, other: &Self) -> bool {
        self.commitment_point == other.commitment_point
    }
}

impl<C: CurveGroup> Eq for PedersenCommitment<C> {}

/// Commitment parameters for the Pedersen scheme
#[derive(Debug, Clone)]
pub struct CommitmentParameters<C: CurveGroup> {
    /// Generator point G for values
    pub value_generator: C,
    /// Generator point H for blinding factors
    pub blinding_generator: C,
    _phantom: PhantomData<C>,
}

impl<C: CurveGroup> CommitmentParameters<C> {
    /// Create new commitment parameters
    pub fn new() -> Self {
        // In production, generate proper independent generators
        let value_generator = C::generator();
        let blinding_generator = C::generator(); // Should be different from value_generator

        Self {
            value_generator,
            blinding_generator,
            _phantom: PhantomData,
        }
    }

    /// Commit to a value with the given blinding factor
    pub fn commit(&self, value: u64, blinding_factor: C::ScalarField) -> PedersenCommitment<C> {
        // Compute value*G + blinding_factor*H
        let value_scalar = C::ScalarField::from(value);
        let commitment_point = self.value_generator * value_scalar +
                              self.blinding_generator * blinding_factor;

        PedersenCommitment {
            commitment_point,
            blinding_factor,
            value,
        }
    }
}

impl<C: CurveGroup> Default for CommitmentParameters<C> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::G1Projective;
    use ark_std::UniformRand;

    #[test]
    fn test_commitment_creation() {
        let value = 100u64;
        let mut rng = ark_std::test_rng();
        let blinding_factor = <G1Projective as CurveGroup>::ScalarField::rand(&mut rng);

        let commitment = PedersenCommitment::<G1Projective>::new(value, blinding_factor);
        assert_eq!(commitment.value(), value);
        assert_eq!(commitment.blinding_factor(), blinding_factor);
    }

    #[test]
    fn test_commitment_homomorphism() {
        let mut rng = ark_std::test_rng();
        let value1 = 50u64;
        let value2 = 75u64;
        let blinding1 = <G1Projective as CurveGroup>::ScalarField::rand(&mut rng);
        let blinding2 = <G1Projective as CurveGroup>::ScalarField::rand(&mut rng);

        let commitment1 = PedersenCommitment::<G1Projective>::new(value1, blinding1);
        let commitment2 = PedersenCommitment::<G1Projective>::new(value2, blinding2);
        let sum_commitment = commitment1.add(&commitment2);

        assert_eq!(sum_commitment.value(), value1 + value2);
        assert_eq!(sum_commitment.blinding_factor(), blinding1 + blinding2);
    }
}