//! # Q-Recursive-Proofs: Post-Quantum Recursive SNARKs
//!
//! This crate implements recursive post-quantum SNARKs for eliminating weak subjectivity
//! in Q-NarwhalKnight's BFT consensus. It enables trustless light client verification
//! of the entire blockchain history in constant time (~10ms).
//!
//! ## Key Features
//!
//! - **Recursive Verification**: Each epoch proof verifies the previous epoch's proof
//! - **Post-Quantum Security**: Based on LatticeGuard (RLWE) - quantum-resistant
//! - **Decentralized Proving**: P2P network of competing provers via libp2p
//! - **Constant-Time Bootstrap**: New nodes verify entire history in ~10ms
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │              RECURSIVE PROOF CHAIN (IVC)                         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  Epoch N-1         Epoch N           Light Client                │
//! │  ┌─────────┐       ┌─────────┐       ┌─────────────────┐        │
//! │  │ π(N-1)  │──────▶│ π(N)    │──────▶│ Verify(πN) =    │        │
//! │  │         │       │         │       │ 10ms, O(1)      │        │
//! │  └─────────┘       └─────────┘       └─────────────────┘        │
//! │       ↑                 │                                        │
//! │       │                 │                                        │
//! │  Contains:          Contains:                                    │
//! │  - BFT sigs         - Verify(π(N-1))                             │
//! │  - State trans      - BFT sigs for N                             │
//! │  - Block hashes     - State transition N                         │
//! │                     - New state root                             │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Modules
//!
//! - `gadgets`: Circuit gadgets (Poseidon hash, Dilithium verification, Merkle trees)
//! - `circuits`: High-level circuits (BFT signatures, state transitions, epoch transitions)
//! - `protocol`: libp2p protocol for decentralized proof generation
//! - `light_client`: Light client implementation for trustless bootstrap
//!
//! ## q-ivc boundary
//!
//! The production epoch transition proof system is the LatticeGuard
//! `ArithmeticCircuit` path in this crate. `EpochPublicInputs` and its canonical
//! scalar encoding are the shared wire format for prover nodes, peer verifiers,
//! and light clients. Do not add another recursive-proof public-input encoding in
//! this crate or in `q-ivc`; future bridges from `q-ivc` must pass through the
//! adapter boundary specified in
//! `docs/adr/2026-05-18-q-ivc-q-recursive-proofs-adapter.md`.

pub mod circuits;
pub mod gadgets;
pub mod ivc_adapter;
pub mod light_client;
pub mod protocol;

// Re-exports
pub use circuits::{
    BFTSignatureCircuit, EpochTransitionCircuit, LatticeGuardVerifierCircuit,
    StateTransitionCircuit,
};
pub use gadgets::{DilithiumVerifierGadget, MerkleTreeGadget, PoseidonGadget};
pub use ivc_adapter::{decode_public_inputs, encode_public_inputs};
pub use light_client::LightClient;
pub use protocol::{EpochProofSubmission, EpochProofTask, ProverNode};

use q_lattice_guard::{ArithmeticCircuit, LatticeGuardProof, R1CSConstraint, Scalar};
use serde::{Deserialize, Serialize};

/// Public inputs for an epoch proof
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EpochPublicInputs {
    /// Previous epoch's state root
    pub previous_state_root: [u8; 32],

    /// Current epoch's state root (after applying this epoch)
    pub current_state_root: [u8; 32],

    /// Epoch number
    pub epoch: u64,

    /// Block height range covered by this epoch
    pub height_range: (u64, u64),

    /// Hash of validator set for this epoch
    pub validator_set_hash: [u8; 32],

    /// Number of valid BFT signatures (must be >= 2f+1)
    pub signature_count: u32,

    /// Timestamp of last block in epoch
    pub epoch_end_timestamp: u64,
}

impl EpochPublicInputs {
    /// Convert to scalar array for proof verification
    pub fn to_scalars(&self) -> Vec<Scalar> {
        let mut scalars = Vec::with_capacity(16);

        // State roots (8 scalars each, 32 bytes / 4 bytes per scalar)
        for chunk in self.previous_state_root.chunks(4) {
            scalars.push(u32::from_le_bytes(chunk.try_into().unwrap()) as Scalar);
        }
        for chunk in self.current_state_root.chunks(4) {
            scalars.push(u32::from_le_bytes(chunk.try_into().unwrap()) as Scalar);
        }

        // Epoch and height range
        scalars.push(self.epoch);
        scalars.push(self.height_range.0);
        scalars.push(self.height_range.1);

        // Validator set hash (8 scalars)
        for chunk in self.validator_set_hash.chunks(4) {
            scalars.push(u32::from_le_bytes(chunk.try_into().unwrap()) as Scalar);
        }

        // Signature count and timestamp
        scalars.push(self.signature_count as Scalar);
        scalars.push(self.epoch_end_timestamp);

        scalars
    }

    /// Create from scalar array
    pub fn from_scalars(scalars: &[Scalar]) -> Option<Self> {
        if scalars.len() < 28 {
            return None;
        }

        let mut previous_state_root = [0u8; 32];
        let mut current_state_root = [0u8; 32];
        let mut validator_set_hash = [0u8; 32];

        // Decode state roots
        for (i, scalar) in scalars[0..8].iter().enumerate() {
            previous_state_root[i * 4..(i + 1) * 4]
                .copy_from_slice(&(*scalar as u32).to_le_bytes());
        }
        for (i, scalar) in scalars[8..16].iter().enumerate() {
            current_state_root[i * 4..(i + 1) * 4]
                .copy_from_slice(&(*scalar as u32).to_le_bytes());
        }

        let epoch = scalars[16];
        let height_range = (scalars[17], scalars[18]);

        for (i, scalar) in scalars[19..27].iter().enumerate() {
            validator_set_hash[i * 4..(i + 1) * 4]
                .copy_from_slice(&(*scalar as u32).to_le_bytes());
        }

        let signature_count = scalars[27] as u32;
        let epoch_end_timestamp = scalars.get(28).copied().unwrap_or(0);

        Some(Self {
            previous_state_root,
            current_state_root,
            epoch,
            height_range,
            validator_set_hash,
            signature_count,
            epoch_end_timestamp,
        })
    }
}

/// Complete epoch proof (recursive SNARK + public inputs)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpochProof {
    /// The recursive SNARK proof
    pub proof: LatticeGuardProof,

    /// Public inputs (verifiable without the proof)
    pub public_inputs: EpochPublicInputs,

    /// Proof metadata
    pub metadata: EpochProofMetadata,
}

/// Metadata for an epoch proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpochProofMetadata {
    /// Protocol version
    pub version: u32,

    /// Prover peer ID
    pub prover_peer_id: Option<String>,

    /// Proving time in milliseconds
    pub proving_time_ms: u64,

    /// Hardware used (optional)
    pub hardware_info: Option<String>,

    /// Timestamp when proof was created
    pub created_at: u64,
}

/// Configuration for recursive proof generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecursiveProofConfig {
    /// Security level for LatticeGuard
    pub security_level: q_lattice_guard::SecurityLevel,

    /// Maximum epoch size (blocks per epoch)
    pub max_epoch_blocks: usize,

    /// Maximum validators per epoch
    pub max_validators: usize,

    /// Byzantine fault threshold (f)
    pub byzantine_threshold: usize,

    /// Enable GPU acceleration
    pub use_gpu: bool,

    /// Number of parallel proving threads
    pub parallel_threads: usize,
}

impl Default for RecursiveProofConfig {
    fn default() -> Self {
        Self {
            security_level: q_lattice_guard::SecurityLevel::PQ128,
            max_epoch_blocks: 10000,
            max_validators: 100,
            byzantine_threshold: 33,
            use_gpu: false,
            parallel_threads: num_cpus::get(),
        }
    }
}

/// Wire index tracker for circuit construction
#[derive(Clone, Debug, Default)]
pub struct WireAllocator {
    /// Next available wire index
    next_wire: usize,
    /// Public input wires
    pub public_inputs: Vec<usize>,
    /// Witness wires
    pub witness: Vec<usize>,
}

impl WireAllocator {
    /// Create new wire allocator
    pub fn new() -> Self {
        Self {
            next_wire: 0,
            public_inputs: Vec::new(),
            witness: Vec::new(),
        }
    }

    /// Allocate a new public input wire
    pub fn alloc_public_input(&mut self) -> usize {
        let wire = self.next_wire;
        self.next_wire += 1;
        self.public_inputs.push(wire);
        wire
    }

    /// Allocate a new witness wire
    pub fn alloc_witness(&mut self) -> usize {
        let wire = self.next_wire;
        self.next_wire += 1;
        self.witness.push(wire);
        wire
    }

    /// Allocate multiple witness wires
    pub fn alloc_witness_array(&mut self, count: usize) -> Vec<usize> {
        (0..count).map(|_| self.alloc_witness()).collect()
    }

    /// Get total wire count
    pub fn wire_count(&self) -> usize {
        self.next_wire
    }
}

/// Constraint builder helper for circuit construction
pub struct ConstraintBuilder {
    /// Accumulated constraints
    constraints: Vec<R1CSConstraint>,
    /// Wire allocator
    pub allocator: WireAllocator,
    /// Modulus for arithmetic
    modulus: Scalar,
}

impl ConstraintBuilder {
    /// Create new constraint builder
    pub fn new(modulus: Scalar) -> Self {
        Self {
            constraints: Vec::new(),
            allocator: WireAllocator::new(),
            modulus,
        }
    }

    /// Add multiplication constraint: a * b = c
    pub fn add_mul(&mut self, a: usize, b: usize, c: usize) {
        self.constraints.push(R1CSConstraint {
            a: vec![(a, 1)],
            b: vec![(b, 1)],
            c: vec![(c, 1)],
        });
    }

    /// Add linear combination constraint: sum(coeffs[i] * wires[i]) = output
    pub fn add_linear_combination(
        &mut self,
        wires: &[(usize, Scalar)],
        output: usize,
    ) {
        // Encode as: (sum of wires) * 1 = output
        self.constraints.push(R1CSConstraint {
            a: wires.to_vec(),
            b: vec![(0, 1)], // Wire 0 is always 1
            c: vec![(output, 1)],
        });
    }

    /// Add constant constraint: wire = constant
    pub fn add_constant(&mut self, wire: usize, constant: Scalar) {
        // Encode as: constant * 1 = wire
        self.constraints.push(R1CSConstraint {
            a: vec![(0, constant)], // Wire 0 * constant
            b: vec![(0, 1)],        // * 1
            c: vec![(wire, 1)],     // = wire
        });
    }

    /// Add equality constraint: a = b
    pub fn add_equality(&mut self, a: usize, b: usize) {
        // Encode as: a * 1 = b
        self.constraints.push(R1CSConstraint {
            a: vec![(a, 1)],
            b: vec![(0, 1)],
            c: vec![(b, 1)],
        });
    }

    /// Add boolean constraint: wire * (1 - wire) = 0 (wire is 0 or 1)
    pub fn add_boolean(&mut self, wire: usize) {
        let one_minus_wire = self.allocator.alloc_witness();

        // one_minus_wire = 1 - wire
        self.constraints.push(R1CSConstraint {
            a: vec![(0, 1)],                                // 1
            b: vec![(0, 1)],                                // * 1
            c: vec![(wire, 1), (one_minus_wire, 1)],        // = wire + one_minus_wire
        });

        // wire * one_minus_wire = 0
        self.constraints.push(R1CSConstraint {
            a: vec![(wire, 1)],
            b: vec![(one_minus_wire, 1)],
            c: vec![],  // = 0
        });
    }

    /// Add AND gate: c = a AND b (where a, b are boolean)
    pub fn add_and(&mut self, a: usize, b: usize) -> usize {
        let c = self.allocator.alloc_witness();
        self.add_mul(a, b, c);
        c
    }

    /// Add XOR gate: c = a XOR b (where a, b are boolean)
    pub fn add_xor(&mut self, a: usize, b: usize) -> usize {
        // XOR: a + b - 2*a*b
        let ab = self.add_and(a, b);
        let two_ab = self.allocator.alloc_witness();
        let c = self.allocator.alloc_witness();

        // two_ab = 2 * ab
        self.add_linear_combination(&[(ab, 2)], two_ab);

        // c = a + b - two_ab
        self.constraints.push(R1CSConstraint {
            a: vec![(a, 1), (b, 1)],
            b: vec![(0, 1)],
            c: vec![(c, 1), (two_ab, 1)],
        });

        c
    }

    /// Build final arithmetic circuit
    pub fn build(self) -> ArithmeticCircuit {
        ArithmeticCircuit {
            num_constraints: self.constraints.len(),
            num_public_inputs: self.allocator.public_inputs.len(),
            num_witness: self.allocator.witness.len(),
            constraints: self.constraints,
        }
    }

    /// Get constraint count
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Get modulus
    pub fn modulus(&self) -> Scalar {
        self.modulus
    }
}

/// Estimate constraint counts for various circuit components
pub mod constraint_estimates {
    /// Poseidon hash (single invocation)
    pub const POSEIDON_HASH: usize = 300;

    /// Dilithium signature verification
    pub const DILITHIUM_SIGNATURE: usize = 100_000;

    /// Merkle tree verification (per level)
    pub const MERKLE_LEVEL: usize = 600;

    /// LatticeGuard verifier circuit
    pub const LATTICE_GUARD_VERIFIER: usize = 100_000;

    /// State root computation
    pub const STATE_ROOT: usize = 50_000;

    /// Single block validation
    pub const BLOCK_VALIDATION: usize = 5_000;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epoch_public_inputs_roundtrip() {
        let inputs = EpochPublicInputs {
            previous_state_root: [1u8; 32],
            current_state_root: [2u8; 32],
            epoch: 42,
            height_range: (100, 200),
            validator_set_hash: [3u8; 32],
            signature_count: 67,
            epoch_end_timestamp: 1700000000,
        };

        let scalars = inputs.to_scalars();
        let recovered = EpochPublicInputs::from_scalars(&scalars).unwrap();

        assert_eq!(inputs, recovered);
    }

    #[test]
    fn test_constraint_builder_basic() {
        let mut builder = ConstraintBuilder::new(1 << 32);

        let a = builder.allocator.alloc_public_input();
        let b = builder.allocator.alloc_public_input();
        let c = builder.allocator.alloc_witness();

        builder.add_mul(a, b, c);

        let circuit = builder.build();
        assert_eq!(circuit.num_constraints, 1);
        assert_eq!(circuit.num_public_inputs, 2);
        assert_eq!(circuit.num_witness, 1);
    }

    #[test]
    fn test_wire_allocator() {
        let mut alloc = WireAllocator::new();

        let pi1 = alloc.alloc_public_input();
        let pi2 = alloc.alloc_public_input();
        let w1 = alloc.alloc_witness();

        assert_eq!(pi1, 0);
        assert_eq!(pi2, 1);
        assert_eq!(w1, 2);
        assert_eq!(alloc.wire_count(), 3);
    }
}
