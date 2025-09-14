//! Groth16 zk-SNARK implementation for Q-NarwhalKnight
//!
//! Groth16 provides the most efficient verification (2 pairing operations)
//! making it ideal for blockchain applications where verification cost matters.

use anyhow::Result;
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_groth16::{Groth16, PreparedVerifyingKey, Proof, ProvingKey, VerifyingKey};
use ark_relations::r1cs::{ConstraintSynthesizer, SynthesisError};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_snark::SNARK;
use ark_std::{
    marker::PhantomData,
    rand::{CryptoRng, RngCore},
    vec::Vec,
};
use serde::{Deserialize, Serialize};

use crate::{circuits::ArithmeticCircuit, SNARKError};

/// Groth16 SNARK implementation
pub struct Groth16SNARK<E: Pairing> {
    _phantom: PhantomData<E>,
}

/// Groth16 proving key
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct Groth16ProvingKey<E: Pairing> {
    pub ark_proving_key: ProvingKey<E>,
}

/// Groth16 verifying key  
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct Groth16VerifyingKey<E: Pairing> {
    pub ark_verifying_key: VerifyingKey<E>,
    pub prepared_vk: PreparedVerifyingKey<E>,
}

/// Groth16 proof
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct Groth16Proof<E: Pairing> {
    pub ark_proof: Proof<E>,
    pub public_inputs: Vec<E::ScalarField>,
}

/// Groth16 circuit wrapper
pub struct Groth16Circuit<F: PrimeField> {
    pub circuit: ArithmeticCircuit<F>,
    pub witness: Option<Vec<F>>,
}

impl<E: Pairing> Groth16SNARK<E> {
    /// Create new Groth16 SNARK system
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Perform trusted setup for a circuit
    pub fn setup<C, R>(
        circuit: C,
        rng: &mut R,
    ) -> Result<(Groth16ProvingKey<E>, Groth16VerifyingKey<E>)>
    where
        C: ConstraintSynthesizer<E::ScalarField>,
        R: RngCore + CryptoRng,
    {
        let pk =
            Groth16::<E>::generate_random_parameters_with_reduction(circuit, rng).map_err(|e| {
                anyhow::anyhow!(SNARKError::SetupFailed(format!(
                    "Groth16 setup failed: {:?}",
                    e
                )))
            })?;
        let vk = pk.vk.clone();

        let prepared_vk = PreparedVerifyingKey::from(vk.clone());

        Ok((
            Groth16ProvingKey {
                ark_proving_key: pk,
            },
            Groth16VerifyingKey {
                ark_verifying_key: vk,
                prepared_vk,
            },
        ))
    }

    /// Generate proof for circuit
    pub fn prove<C, R>(
        proving_key: &Groth16ProvingKey<E>,
        circuit: C,
        rng: &mut R,
    ) -> Result<Groth16Proof<E>>
    where
        C: ConstraintSynthesizer<E::ScalarField>,
        R: RngCore + CryptoRng,
    {
        // For now, use empty public inputs - this would be populated from circuit
        let public_inputs: Vec<E::ScalarField> = vec![];

        // Generate proof
        let ark_proof =
            Groth16::<E>::prove(&proving_key.ark_proving_key, circuit, rng).map_err(|e| {
                anyhow::anyhow!(SNARKError::ProvingFailed(format!(
                    "Groth16 proving failed: {:?}",
                    e
                )))
            })?;

        Ok(Groth16Proof {
            ark_proof,
            public_inputs,
        })
    }

    /// Verify proof
    pub fn verify(verifying_key: &Groth16VerifyingKey<E>, proof: &Groth16Proof<E>) -> Result<bool> {
        let result = Groth16::<E>::verify_with_processed_vk(
            &verifying_key.prepared_vk,
            &proof.public_inputs,
            &proof.ark_proof,
        )
        .map_err(|e| {
            anyhow::anyhow!(SNARKError::VerificationFailed(format!(
                "Groth16 verification failed: {:?}",
                e
            )))
        })?;

        Ok(result)
    }

    /// Batch verify multiple proofs (more efficient than individual verification)
    pub fn batch_verify(
        verifying_key: &Groth16VerifyingKey<E>,
        proofs: &[Groth16Proof<E>],
    ) -> Result<bool> {
        if proofs.is_empty() {
            return Ok(true);
        }

        // Extract public inputs and ark proofs
        let public_inputs: Vec<_> = proofs.iter().map(|p| p.public_inputs.as_slice()).collect();
        let ark_proofs: Vec<_> = proofs.iter().map(|p| &p.ark_proof).collect();

        // Use ark_groth16 batch verification if available
        // For now, verify each proof individually (can be optimized later)
        for (public_input, ark_proof) in public_inputs.iter().zip(ark_proofs.iter()) {
            let result = Groth16::<E>::verify_with_processed_vk(
                &verifying_key.prepared_vk,
                public_input,
                ark_proof,
            )
            .map_err(|e| {
                anyhow::anyhow!(SNARKError::VerificationFailed(format!(
                    "Batch verification failed: {:?}",
                    e
                )))
            })?;

            if !result {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Serialize proving key
    pub fn serialize_proving_key(pk: &Groth16ProvingKey<E>) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        pk.ark_proving_key
            .serialize_uncompressed(&mut bytes)
            .map_err(|e| SNARKError::Serialization(format!("PK serialization failed: {:?}", e)))?;
        Ok(bytes)
    }

    /// Deserialize proving key
    pub fn deserialize_proving_key(bytes: &[u8]) -> Result<Groth16ProvingKey<E>> {
        let ark_proving_key = ProvingKey::<E>::deserialize_uncompressed(bytes).map_err(|e| {
            SNARKError::Serialization(format!("PK deserialization failed: {:?}", e))
        })?;
        Ok(Groth16ProvingKey { ark_proving_key })
    }

    /// Serialize verifying key
    pub fn serialize_verifying_key(vk: &Groth16VerifyingKey<E>) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        vk.ark_verifying_key
            .serialize_uncompressed(&mut bytes)
            .map_err(|e| SNARKError::Serialization(format!("VK serialization failed: {:?}", e)))?;
        Ok(bytes)
    }

    /// Deserialize verifying key
    pub fn deserialize_verifying_key(bytes: &[u8]) -> Result<Groth16VerifyingKey<E>> {
        let ark_verifying_key =
            VerifyingKey::<E>::deserialize_uncompressed(bytes).map_err(|e| {
                SNARKError::Serialization(format!("VK deserialization failed: {:?}", e))
            })?;

        let prepared_vk = PreparedVerifyingKey::from(ark_verifying_key.clone());

        Ok(Groth16VerifyingKey {
            ark_verifying_key,
            prepared_vk,
        })
    }

    /// Serialize proof
    pub fn serialize_proof(proof: &Groth16Proof<E>) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        proof.serialize_uncompressed(&mut bytes).map_err(|e| {
            SNARKError::Serialization(format!("Proof serialization failed: {:?}", e))
        })?;
        Ok(bytes)
    }

    /// Deserialize proof
    pub fn deserialize_proof(bytes: &[u8]) -> Result<Groth16Proof<E>> {
        let proof = Groth16Proof::<E>::deserialize_uncompressed(bytes).map_err(|e| {
            SNARKError::Serialization(format!("Proof deserialization failed: {:?}", e))
        })?;
        Ok(proof)
    }
}

impl<F: PrimeField> ConstraintSynthesizer<F> for Groth16Circuit<F> {
    fn generate_constraints(
        self,
        cs: ark_relations::r1cs::ConstraintSystemRef<F>,
    ) -> std::result::Result<(), SynthesisError> {
        // Convert our arithmetic circuit to R1CS constraints
        self.circuit
            .generate_r1cs_constraints(cs, &self.witness)
            .map_err(|_| SynthesisError::Unsatisfiable)?;
        Ok(())
    }
}

/// Groth16-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Groth16Config {
    /// Enable preprocessing optimizations
    pub preprocessing: bool,
    /// Enable batch verification
    pub batch_verification: bool,
    /// Maximum batch size for verification
    pub max_batch_size: usize,
}

impl Default for Groth16Config {
    fn default() -> Self {
        Self {
            preprocessing: true,
            batch_verification: true,
            max_batch_size: 100,
        }
    }
}

/// Groth16 utilities
pub struct Groth16Utils;

impl Groth16Utils {
    /// Estimate proof generation time based on constraint count
    pub fn estimate_proving_time(num_constraints: usize) -> std::time::Duration {
        // Rough estimates based on typical hardware
        let base_time_ms = 100; // Base overhead
        let per_constraint_ns = 50; // Nanoseconds per constraint

        let total_ms = base_time_ms + (num_constraints * per_constraint_ns) / 1_000_000;
        std::time::Duration::from_millis(total_ms as u64)
    }

    /// Estimate verification time (constant for Groth16)
    pub fn estimate_verification_time() -> std::time::Duration {
        std::time::Duration::from_millis(2) // ~2ms for 2 pairing operations
    }

    /// Estimate proof size (constant for Groth16)
    pub fn estimate_proof_size() -> usize {
        256 // ~256 bytes for 2 G1 elements + 1 G2 element
    }

    /// Check if circuit size is suitable for Groth16
    pub fn is_suitable_circuit_size(num_constraints: usize) -> bool {
        // Groth16 is most efficient for small to medium circuits
        num_constraints <= 1_000_000
    }

    /// Generate random test circuit for benchmarking
    pub fn generate_test_circuit<F: PrimeField>(num_constraints: usize) -> ArithmeticCircuit<F> {
        ArithmeticCircuit::new(num_constraints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits::ArithmeticCircuit;
    use ark_bn254::{Bn254, Fr};
    use ark_std::test_rng;

    #[test]
    fn test_groth16_setup_prove_verify() {
        let mut rng = test_rng();

        // Create simple test circuit
        let circuit = ArithmeticCircuit::<Fr>::new(10);
        let witness = vec![Fr::from(1u64); 5];

        let groth16_circuit = Groth16Circuit {
            circuit: circuit.clone(),
            witness: Some(witness.clone()),
        };

        // Setup
        let (pk, vk) = Groth16SNARK::<Bn254>::setup(groth16_circuit.clone(), &mut rng)
            .expect("Setup should succeed");

        // Prove
        let proof = Groth16SNARK::<Bn254>::prove(&pk, groth16_circuit, &mut rng)
            .expect("Proving should succeed");

        // Verify
        let is_valid =
            Groth16SNARK::<Bn254>::verify(&vk, &proof).expect("Verification should succeed");

        assert!(is_valid, "Proof should be valid");
    }

    #[test]
    fn test_groth16_batch_verification() {
        let mut rng = test_rng();

        // Create test circuit
        let circuit = ArithmeticCircuit::<Fr>::new(5);
        let witness = vec![Fr::from(1u64); 3];

        let groth16_circuit = Groth16Circuit {
            circuit,
            witness: Some(witness),
        };

        // Setup
        let (pk, vk) = Groth16SNARK::<Bn254>::setup(groth16_circuit.clone(), &mut rng)
            .expect("Setup should succeed");

        // Generate multiple proofs
        let mut proofs = Vec::new();
        for _ in 0..3 {
            let proof = Groth16SNARK::<Bn254>::prove(&pk, groth16_circuit.clone(), &mut rng)
                .expect("Proving should succeed");
            proofs.push(proof);
        }

        // Batch verify
        let all_valid = Groth16SNARK::<Bn254>::batch_verify(&vk, &proofs)
            .expect("Batch verification should succeed");

        assert!(all_valid, "All proofs should be valid");
    }

    #[test]
    fn test_groth16_serialization() {
        let mut rng = test_rng();

        // Create test circuit
        let circuit = ArithmeticCircuit::<Fr>::new(5);
        let witness = vec![Fr::from(1u64); 3];

        let groth16_circuit = Groth16Circuit {
            circuit,
            witness: Some(witness),
        };

        // Setup
        let (pk, vk) = Groth16SNARK::<Bn254>::setup(groth16_circuit.clone(), &mut rng)
            .expect("Setup should succeed");

        // Test proving key serialization
        let pk_bytes = Groth16SNARK::<Bn254>::serialize_proving_key(&pk)
            .expect("PK serialization should succeed");
        let pk_restored = Groth16SNARK::<Bn254>::deserialize_proving_key(&pk_bytes)
            .expect("PK deserialization should succeed");

        // Test verifying key serialization
        let vk_bytes = Groth16SNARK::<Bn254>::serialize_verifying_key(&vk)
            .expect("VK serialization should succeed");
        let vk_restored = Groth16SNARK::<Bn254>::deserialize_verifying_key(&vk_bytes)
            .expect("VK deserialization should succeed");

        // Generate and serialize proof
        let proof = Groth16SNARK::<Bn254>::prove(&pk_restored, groth16_circuit, &mut rng)
            .expect("Proving should succeed");

        let proof_bytes = Groth16SNARK::<Bn254>::serialize_proof(&proof)
            .expect("Proof serialization should succeed");
        let proof_restored = Groth16SNARK::<Bn254>::deserialize_proof(&proof_bytes)
            .expect("Proof deserialization should succeed");

        // Verify with restored keys
        let is_valid = Groth16SNARK::<Bn254>::verify(&vk_restored, &proof_restored)
            .expect("Verification should succeed");

        assert!(
            is_valid,
            "Proof should be valid after serialization round-trip"
        );
    }

    #[test]
    fn test_groth16_utils() {
        // Test time estimation
        let proving_time = Groth16Utils::estimate_proving_time(10000);
        assert!(proving_time.as_millis() > 0);

        let verification_time = Groth16Utils::estimate_verification_time();
        assert_eq!(verification_time.as_millis(), 2);

        // Test size estimation
        let proof_size = Groth16Utils::estimate_proof_size();
        assert_eq!(proof_size, 256);

        // Test circuit size suitability
        assert!(Groth16Utils::is_suitable_circuit_size(1000));
        assert!(Groth16Utils::is_suitable_circuit_size(1_000_000));
        assert!(!Groth16Utils::is_suitable_circuit_size(2_000_000));
    }
}
