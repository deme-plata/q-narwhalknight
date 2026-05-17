//! Epoch Transition Circuit
//!
//! The complete epoch transition circuit that combines:
//! - Recursive proof verification (LatticeGuardVerifierCircuit)
//! - BFT signature verification (BFTSignatureCircuit)
//! - State transition verification (StateTransitionCircuit)
//!
//! This is the top-level circuit that enables IVC for eliminating weak subjectivity.
//!
//! ## Constraint Breakdown
//!
//! | Component | Constraints |
//! |-----------|-------------|
//! | Recursive Verifier | ~100,000 |
//! | BFT Signatures | ~150,000 |
//! | State Transition | ~200,000 |
//! | Glue Logic | ~50,000 |
//! | **Total** | **~500,000** |

use super::bft_signature::{
    AggregatedBFTSignaturesWires, BFTSignatureCircuit, BFTSignatureConfig, ValidatorSetWires,
};
use super::lattice_verifier::{
    allocate_proof_wires, LatticeGuardProofWires, LatticeGuardVerifierCircuit, VerifierConfig,
};
use super::state_transition::{
    EpochBlocksWires, StateTransitionCircuit, StateTransitionConfig, StateTransitionWitnessWires,
};
use crate::gadgets::poseidon::PoseidonGadget;
use crate::{ConstraintBuilder, EpochPublicInputs};
use q_lattice_guard::{ArithmeticCircuit, R1CSConstraint, Scalar};
use serde::{Deserialize, Serialize};

/// Epoch transition circuit configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpochTransitionConfig {
    /// Verifier configuration
    pub verifier_config: VerifierConfig,
    /// BFT signature configuration
    pub bft_config: BFTSignatureConfig,
    /// State transition configuration
    pub state_config: StateTransitionConfig,
    /// Is this the genesis epoch (no previous proof)
    pub is_genesis: bool,
}

impl Default for EpochTransitionConfig {
    fn default() -> Self {
        Self {
            verifier_config: VerifierConfig::default(),
            bft_config: BFTSignatureConfig::default(),
            state_config: StateTransitionConfig::default(),
            is_genesis: false,
        }
    }
}

impl EpochTransitionConfig {
    /// Create a "light mode" configuration for faster SRS generation
    ///
    /// This reduces constraint count from ~3M to ~200K by:
    /// - Reducing validators from 100 to 20
    /// - Reducing blocks per epoch for proofs from 1000 to 100
    /// - Using smaller verifier config
    ///
    /// Ideal for light client bootstrap and testnet environments.
    pub fn light_mode() -> Self {
        Self {
            verifier_config: VerifierConfig::light(),
            bft_config: BFTSignatureConfig::light(),
            state_config: StateTransitionConfig::light(),
            is_genesis: false,
        }
    }

    /// Create genesis config with light mode settings
    pub fn genesis_light() -> Self {
        let mut config = Self::light_mode();
        config.is_genesis = true;
        config
    }
}

/// All wires needed for epoch transition proof
#[derive(Clone, Debug)]
pub struct EpochTransitionWires {
    // === Previous Epoch Proof (for recursion) ===
    /// Previous epoch's proof (verified recursively)
    pub previous_proof: Option<LatticeGuardProofWires>,
    /// Previous epoch's public inputs
    pub previous_public_inputs: Option<Vec<usize>>,

    // === Current Epoch Data ===
    /// Current epoch number
    pub epoch: usize,
    /// Block height range
    pub height_start: usize,
    pub height_end: usize,

    // === BFT Data ===
    /// Validator set
    pub validator_set: ValidatorSetWires,
    /// Aggregated signatures
    pub signatures: AggregatedBFTSignaturesWires,
    /// Message being signed (epoch block commitment)
    pub signed_message: [usize; 8],

    // === State Transition Data ===
    /// Epoch blocks
    pub blocks: EpochBlocksWires,
    /// State transition witness
    pub state_witness: StateTransitionWitnessWires,

    // === Public Outputs ===
    /// Previous state root (public input)
    pub prev_state_root: [usize; 8],
    /// New state root (public output)
    pub new_state_root: [usize; 8],
    /// Validator set hash (for next epoch)
    pub validator_set_hash: [usize; 8],
}

/// Epoch transition circuit (top-level)
pub struct EpochTransitionCircuit {
    /// Configuration
    config: EpochTransitionConfig,
    /// Recursive verifier
    verifier: LatticeGuardVerifierCircuit,
    /// BFT signature circuit
    bft: BFTSignatureCircuit,
    /// State transition circuit
    state: StateTransitionCircuit,
    /// Poseidon for hashing
    poseidon: PoseidonGadget,
}

impl EpochTransitionCircuit {
    /// Create new epoch transition circuit
    pub fn new(config: EpochTransitionConfig) -> Self {
        Self {
            verifier: LatticeGuardVerifierCircuit::new(config.verifier_config.clone()),
            bft: BFTSignatureCircuit::new(config.bft_config.clone()),
            state: StateTransitionCircuit::new(config.state_config.clone()),
            poseidon: PoseidonGadget::new(16),
            config,
        }
    }

    /// Create for genesis epoch (no recursion needed)
    pub fn genesis() -> Self {
        let mut config = EpochTransitionConfig::default();
        config.is_genesis = true;
        Self::new(config)
    }

    /// Synthesize the complete epoch transition circuit
    ///
    /// Returns a wire that is 1 if the epoch transition is valid.
    pub fn synthesize(
        &self,
        builder: &mut ConstraintBuilder,
        wires: &EpochTransitionWires,
    ) -> usize {
        // ============================================================
        // SECTION 1: Recursive Proof Verification
        // ============================================================
        // Verify the previous epoch's proof (if not genesis)

        let recursive_valid = if self.config.is_genesis {
            // Genesis epoch: no previous proof to verify
            let valid = builder.allocator.alloc_witness();
            builder.add_constant(valid, 1);
            valid
        } else {
            match (&wires.previous_proof, &wires.previous_public_inputs) {
                (Some(proof), Some(inputs)) => {
                    self.verifier.synthesize(builder, proof, inputs)
                }
                _ => {
                    // Missing proof/inputs for non-genesis = invalid
                    let invalid = builder.allocator.alloc_witness();
                    builder.add_constant(invalid, 0);
                    invalid
                }
            }
        };

        // ============================================================
        // SECTION 2: State Root Continuity
        // ============================================================
        // Previous epoch's new_state_root should equal this epoch's prev_state_root

        let continuity_valid = if !self.config.is_genesis {
            self.verify_state_continuity(builder, wires)
        } else {
            let valid = builder.allocator.alloc_witness();
            builder.add_constant(valid, 1);
            valid
        };

        // ============================================================
        // SECTION 3: BFT Signature Verification
        // ============================================================
        // Verify that 2f+1 validators signed the epoch

        let bft_valid = self.bft.synthesize_aggregated(
            builder,
            &wires.signed_message,
            &wires.validator_set,
            &wires.signatures,
        );

        // ============================================================
        // SECTION 4: Signed Message Correctness
        // ============================================================
        // Verify that the signed message is correctly computed

        let message_valid = self.verify_signed_message(builder, wires);

        // ============================================================
        // SECTION 5: State Transition Verification
        // ============================================================
        // Verify the state transition is valid

        let state_valid = self.state.synthesize(builder, &wires.blocks, &wires.state_witness);

        // ============================================================
        // SECTION 6: Public Input Binding
        // ============================================================
        // Verify public inputs/outputs are correctly bound

        let binding_valid = self.verify_public_binding(builder, wires);

        // ============================================================
        // FINAL: Combine all checks
        // ============================================================

        let valid_1 = builder.add_and(recursive_valid, continuity_valid);
        let valid_2 = builder.add_and(valid_1, bft_valid);
        let valid_3 = builder.add_and(valid_2, message_valid);
        let valid_4 = builder.add_and(valid_3, state_valid);
        let valid_final = builder.add_and(valid_4, binding_valid);

        valid_final
    }

    /// Verify state root continuity between epochs
    fn verify_state_continuity(
        &self,
        builder: &mut ConstraintBuilder,
        wires: &EpochTransitionWires,
    ) -> usize {
        // Previous epoch's public inputs should include new_state_root
        // which should equal current epoch's prev_state_root

        if let Some(prev_inputs) = &wires.previous_public_inputs {
            // Public inputs format: [prev_root (8), new_root (8), epoch, ...]
            // We check that prev_inputs[8..16] == wires.prev_state_root

            let mut all_match = builder.allocator.alloc_witness();
            builder.add_constant(all_match, 1);

            for i in 0..8 {
                if i + 8 < prev_inputs.len() {
                    let diff = builder.allocator.alloc_witness();
                    builder.constraints.push(R1CSConstraint {
                        a: vec![(prev_inputs[i + 8], 1)],
                        b: vec![(0, 1)],
                        c: vec![(wires.prev_state_root[i], 1), (diff, 1)],
                    });

                    let is_zero = self.check_is_zero(builder, diff);
                    all_match = builder.add_and(all_match, is_zero);
                }
            }

            all_match
        } else {
            let invalid = builder.allocator.alloc_witness();
            builder.add_constant(invalid, 0);
            invalid
        }
    }

    /// Verify the signed message is correctly computed
    fn verify_signed_message(
        &self,
        builder: &mut ConstraintBuilder,
        wires: &EpochTransitionWires,
    ) -> usize {
        // Compute expected message from epoch data
        let mut message_input = Vec::new();

        // Include epoch number
        message_input.push(wires.epoch);

        // Include height range
        message_input.push(wires.height_start);
        message_input.push(wires.height_end);

        // Include state roots
        message_input.extend_from_slice(&wires.prev_state_root);
        message_input.extend_from_slice(&wires.new_state_root);

        // Include validator set hash
        message_input.extend_from_slice(&wires.validator_set_hash);

        // Hash to get expected message
        let output = self.poseidon.synthesize(builder, &message_input);

        // Compare with claimed signed message
        let mut all_match = builder.allocator.alloc_witness();
        builder.add_constant(all_match, 1);

        for i in 0..8.min(output.len()) {
            let diff = builder.allocator.alloc_witness();
            builder.constraints.push(R1CSConstraint {
                a: vec![(output[i], 1)],
                b: vec![(0, 1)],
                c: vec![(wires.signed_message[i], 1), (diff, 1)],
            });

            let is_zero = self.check_is_zero(builder, diff);
            all_match = builder.add_and(all_match, is_zero);
        }

        all_match
    }

    /// Verify public inputs/outputs are correctly bound
    fn verify_public_binding(
        &self,
        builder: &mut ConstraintBuilder,
        wires: &EpochTransitionWires,
    ) -> usize {
        // Verify state_witness roots match public inputs
        let mut all_valid = builder.allocator.alloc_witness();
        builder.add_constant(all_valid, 1);

        // prev_state_root should match state_witness.prev_state_root
        let prev_match = self.verify_hash_equality(
            builder,
            &wires.prev_state_root,
            &wires.state_witness.prev_state_root,
        );
        all_valid = builder.add_and(all_valid, prev_match);

        // new_state_root should match state_witness.new_state_root
        let new_match = self.verify_hash_equality(
            builder,
            &wires.new_state_root,
            &wires.state_witness.new_state_root,
        );
        all_valid = builder.add_and(all_valid, new_match);

        all_valid
    }

    /// Verify two hash arrays are equal
    fn verify_hash_equality(
        &self,
        builder: &mut ConstraintBuilder,
        a: &[usize; 8],
        b: &[usize; 8],
    ) -> usize {
        let mut all_equal = builder.allocator.alloc_witness();
        builder.add_constant(all_equal, 1);

        for i in 0..8 {
            let diff = builder.allocator.alloc_witness();
            builder.constraints.push(R1CSConstraint {
                a: vec![(a[i], 1)],
                b: vec![(0, 1)],
                c: vec![(b[i], 1), (diff, 1)],
            });

            let is_zero = self.check_is_zero(builder, diff);
            all_equal = builder.add_and(all_equal, is_zero);
        }

        all_equal
    }

    /// Check if value is zero
    fn check_is_zero(&self, builder: &mut ConstraintBuilder, value: usize) -> usize {
        let inverse = builder.allocator.alloc_witness();
        let value_times_inverse = builder.allocator.alloc_witness();
        let is_zero = builder.allocator.alloc_witness();

        builder.add_mul(value, inverse, value_times_inverse);

        builder.constraints.push(R1CSConstraint {
            a: vec![(0, 1)],
            b: vec![(0, 1)],
            c: vec![(is_zero, 1), (value_times_inverse, 1)],
        });

        let should_be_zero = builder.allocator.alloc_witness();
        builder.add_mul(value, is_zero, should_be_zero);
        builder.add_constant(should_be_zero, 0);

        is_zero
    }

    /// Estimate total constraint count
    pub fn estimate_constraints(&self, num_blocks: usize) -> usize {
        let recursive = if self.config.is_genesis {
            0
        } else {
            self.verifier.estimate_constraints()
        };

        let bft = self.bft.estimate_constraints();
        let state = self.state.estimate_constraints(num_blocks);

        // Glue logic: continuity checks, message verification, binding
        let glue = 10_000;

        recursive + bft + state + glue
    }

    /// Build complete circuit
    pub fn build_circuit(&self, num_blocks: usize) -> ArithmeticCircuit {
        let mut builder = ConstraintBuilder::new(1 << 32);

        // Allocate all wires
        let wires = self.allocate_wires(&mut builder, num_blocks);

        // Synthesize circuit
        let _valid = self.synthesize(&mut builder, &wires);

        builder.build()
    }

    /// Allocate all wires needed for the circuit
    fn allocate_wires(
        &self,
        builder: &mut ConstraintBuilder,
        num_blocks: usize,
    ) -> EpochTransitionWires {
        // Previous proof (if not genesis)
        let previous_proof = if !self.config.is_genesis {
            Some(allocate_proof_wires(builder, &self.config.verifier_config))
        } else {
            None
        };

        let previous_public_inputs = if !self.config.is_genesis {
            Some((0..32).map(|_| builder.allocator.alloc_witness()).collect())
        } else {
            None
        };

        // Epoch metadata
        let epoch = builder.allocator.alloc_public_input();
        let height_start = builder.allocator.alloc_public_input();
        let height_end = builder.allocator.alloc_public_input();

        // Validator set
        let validator_set = ValidatorSetWires {
            validator_set_hash: self.allocate_hash_wires(builder),
            public_key_hashes: (0..self.config.bft_config.num_validators)
                .map(|_| self.allocate_hash_wires(builder))
                .collect(),
        };

        // Signatures
        let signatures = super::bft_signature::allocate_aggregated_signature_wires(
            builder,
            &self.config.bft_config,
        );

        // Signed message
        let signed_message = self.allocate_hash_wires(builder);

        // Blocks
        let headers = (0..num_blocks)
            .map(|_| super::state_transition::allocate_block_header_wires(builder, 3, true))
            .collect();

        let blocks = EpochBlocksWires {
            headers,
            block_count: num_blocks,
        };

        // State witness
        let state_witness = StateTransitionWitnessWires {
            prev_state_root: self.allocate_hash_wires(builder),
            new_state_root: self.allocate_hash_wires(builder),
            intermediate_roots: Vec::new(),
        };

        // Public state roots
        let prev_state_root = self.allocate_hash_wires_public(builder);
        let new_state_root = self.allocate_hash_wires_public(builder);
        let validator_set_hash = self.allocate_hash_wires_public(builder);

        EpochTransitionWires {
            previous_proof,
            previous_public_inputs,
            epoch,
            height_start,
            height_end,
            validator_set,
            signatures,
            signed_message,
            blocks,
            state_witness,
            prev_state_root,
            new_state_root,
            validator_set_hash,
        }
    }

    /// Allocate 8-element hash wires (witness)
    fn allocate_hash_wires(&self, builder: &mut ConstraintBuilder) -> [usize; 8] {
        [
            builder.allocator.alloc_witness(),
            builder.allocator.alloc_witness(),
            builder.allocator.alloc_witness(),
            builder.allocator.alloc_witness(),
            builder.allocator.alloc_witness(),
            builder.allocator.alloc_witness(),
            builder.allocator.alloc_witness(),
            builder.allocator.alloc_witness(),
        ]
    }

    /// Allocate 8-element hash wires (public input)
    fn allocate_hash_wires_public(&self, builder: &mut ConstraintBuilder) -> [usize; 8] {
        [
            builder.allocator.alloc_public_input(),
            builder.allocator.alloc_public_input(),
            builder.allocator.alloc_public_input(),
            builder.allocator.alloc_public_input(),
            builder.allocator.alloc_public_input(),
            builder.allocator.alloc_public_input(),
            builder.allocator.alloc_public_input(),
            builder.allocator.alloc_public_input(),
        ]
    }

    /// Get configuration
    pub fn config(&self) -> &EpochTransitionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epoch_transition_circuit_creation() {
        let config = EpochTransitionConfig::default();
        let circuit = EpochTransitionCircuit::new(config);

        let constraints = circuit.estimate_constraints(100);
        println!("Epoch transition (100 blocks) constraints: {}", constraints);

        // Should be around 500K
        assert!(constraints > 200_000);
        assert!(constraints < 1_000_000);
    }

    #[test]
    fn test_genesis_epoch_circuit() {
        let circuit = EpochTransitionCircuit::genesis();

        let constraints = circuit.estimate_constraints(100);
        println!("Genesis epoch (100 blocks) constraints: {}", constraints);

        // Genesis should be cheaper (no recursion)
        assert!(constraints < 500_000);
    }

    #[test]
    fn test_epoch_circuit_build() {
        let config = EpochTransitionConfig {
            bft_config: BFTSignatureConfig {
                num_validators: 10, // Small for testing
                f: 3,
                ..Default::default()
            },
            ..Default::default()
        };

        let circuit = EpochTransitionCircuit::new(config);
        let built = circuit.build_circuit(10); // 10 blocks

        println!(
            "Built circuit: {} constraints, {} public, {} witness",
            built.num_constraints, built.num_public_inputs, built.num_witness
        );

        assert!(built.num_constraints > 0);
        assert!(built.num_public_inputs > 0);
    }
}
