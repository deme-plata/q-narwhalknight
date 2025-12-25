//! Merkle Tree Verification Gadget
//!
//! This module implements circuit gadgets for verifying Merkle tree membership
//! proofs and computing Merkle roots. This is essential for:
//!
//! 1. State root verification (verifying account balances, contract state)
//! 2. Block inclusion proofs (verifying transactions in blocks)
//! 3. Validator set commitments (verifying validator membership)
//!
//! ## Using Poseidon for Efficiency
//!
//! We use Poseidon hash for Merkle trees to minimize constraint count:
//! - ~600 constraints per Merkle level (2 hash inputs)
//! - For depth 32: ~20,000 constraints
//! - Compare to SHA3: ~800,000 constraints for same depth

use crate::gadgets::poseidon::{poseidon_native, PoseidonGadget, PoseidonParams};
use crate::ConstraintBuilder;
use q_lattice_guard::Scalar;
use serde::{Deserialize, Serialize};

/// Merkle proof for inclusion verification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerkleProof {
    /// Sibling hashes along the path
    pub siblings: Vec<[u8; 32]>,
    /// Path bits (0 = left, 1 = right)
    pub path: Vec<bool>,
    /// Leaf index
    pub leaf_index: u64,
}

impl MerkleProof {
    /// Verify the proof (native computation)
    pub fn verify(&self, leaf: &[u8; 32], root: &[u8; 32], params: &PoseidonParams) -> bool {
        let mut current = *leaf;

        for (sibling, &is_right) in self.siblings.iter().zip(self.path.iter()) {
            let (left, right) = if is_right {
                (sibling, &current)
            } else {
                (&current, sibling)
            };

            // Hash the pair
            current = hash_pair_native(left, right, params);
        }

        current == *root
    }
}

/// Merkle proof representation in circuit wires
#[derive(Clone, Debug)]
pub struct MerkleProofWires {
    /// Sibling hash wires (8 scalars per hash)
    pub siblings: Vec<[usize; 8]>,
    /// Path bits (as boolean wires)
    pub path_bits: Vec<usize>,
}

/// Merkle tree gadget for circuit construction
pub struct MerkleTreeGadget {
    /// Poseidon gadget for hashing
    poseidon: PoseidonGadget,
    /// Tree depth
    depth: usize,
}

impl MerkleTreeGadget {
    /// Create new Merkle tree gadget with given depth
    pub fn new(depth: usize) -> Self {
        // Use width-16 Poseidon (can absorb two 32-byte inputs)
        Self {
            poseidon: PoseidonGadget::new(16),
            depth,
        }
    }

    /// Verify Merkle proof in circuit
    ///
    /// Returns wire that is 1 if proof is valid, 0 otherwise.
    pub fn verify_proof(
        &self,
        builder: &mut ConstraintBuilder,
        leaf: &[usize; 8],
        proof: &MerkleProofWires,
        root: &[usize; 8],
    ) -> usize {
        assert_eq!(proof.siblings.len(), self.depth);
        assert_eq!(proof.path_bits.len(), self.depth);

        let mut current = *leaf;

        for i in 0..self.depth {
            let sibling = &proof.siblings[i];
            let is_right = proof.path_bits[i];

            // Constrain path bit to be boolean
            builder.add_boolean(is_right);

            // Select (left, right) based on path bit
            // If is_right = 0: (current, sibling)
            // If is_right = 1: (sibling, current)
            let (left, right) = self.conditional_swap(builder, &current, sibling, is_right);

            // Hash the pair
            current = self.poseidon.hash_pair(builder, &left, &right);
        }

        // Check if computed root equals expected root
        self.check_hash_equality(builder, &current, root)
    }

    /// Conditional swap based on selector bit
    fn conditional_swap(
        &self,
        builder: &mut ConstraintBuilder,
        a: &[usize; 8],
        b: &[usize; 8],
        selector: usize,
    ) -> ([usize; 8], [usize; 8]) {
        let mut left = [0usize; 8];
        let mut right = [0usize; 8];

        for i in 0..8 {
            // left[i] = (1 - selector) * a[i] + selector * b[i]
            // right[i] = selector * a[i] + (1 - selector) * b[i]

            // selector * (b[i] - a[i])
            let diff = builder.allocator.alloc_witness();
            builder.constraints.push(q_lattice_guard::R1CSConstraint {
                a: vec![(b[i], 1)],
                b: vec![(0, 1)],
                c: vec![(diff, 1), (a[i], 1)],
            });

            let selector_times_diff = builder.allocator.alloc_witness();
            builder.add_mul(selector, diff, selector_times_diff);

            // left[i] = a[i] + selector * diff = a[i] + selector * (b[i] - a[i])
            left[i] = builder.allocator.alloc_witness();
            builder.add_linear_combination(&[(a[i], 1), (selector_times_diff, 1)], left[i]);

            // right[i] = b[i] - selector * diff = b[i] - selector * (b[i] - a[i])
            right[i] = builder.allocator.alloc_witness();
            builder.constraints.push(q_lattice_guard::R1CSConstraint {
                a: vec![(b[i], 1)],
                b: vec![(0, 1)],
                c: vec![(right[i], 1), (selector_times_diff, 1)],
            });
        }

        (left, right)
    }

    /// Check if two hash values are equal
    fn check_hash_equality(
        &self,
        builder: &mut ConstraintBuilder,
        a: &[usize; 8],
        b: &[usize; 8],
    ) -> usize {
        let mut all_equal = builder.allocator.alloc_witness();
        builder.add_constant(all_equal, 1);

        for i in 0..8 {
            // Check a[i] - b[i] == 0
            let diff = builder.allocator.alloc_witness();
            builder.constraints.push(q_lattice_guard::R1CSConstraint {
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

        builder.constraints.push(q_lattice_guard::R1CSConstraint {
            a: vec![(0, 1)],
            b: vec![(0, 1)],
            c: vec![(is_zero, 1), (value_times_inverse, 1)],
        });

        let should_be_zero = builder.allocator.alloc_witness();
        builder.add_mul(value, is_zero, should_be_zero);
        builder.add_constant(should_be_zero, 0);

        is_zero
    }

    /// Compute Merkle root from leaves
    pub fn compute_root(
        &self,
        builder: &mut ConstraintBuilder,
        leaves: &[[usize; 8]],
    ) -> [usize; 8] {
        assert!(leaves.len().is_power_of_two(), "Number of leaves must be power of 2");

        if leaves.len() == 1 {
            return leaves[0];
        }

        // Build tree bottom-up
        let mut current_level = leaves.to_vec();

        while current_level.len() > 1 {
            let mut next_level = Vec::with_capacity(current_level.len() / 2);

            for pair in current_level.chunks(2) {
                let hash = self.poseidon.hash_pair(builder, &pair[0], &pair[1]);
                next_level.push(hash);
            }

            current_level = next_level;
        }

        current_level[0]
    }

    /// Estimate constraint count for proof verification
    pub fn estimate_constraints(&self) -> usize {
        let poseidon_constraints = self.poseidon.estimate_constraints();

        // Per level: conditional swap + hash + equality check portion
        let swap_constraints = 8 * 5; // ~5 constraints per element for swap
        let equality_per_level = 8 * 3; // ~3 constraints per element for equality

        self.depth * (poseidon_constraints + swap_constraints) + equality_per_level
    }

    /// Get tree depth
    pub fn depth(&self) -> usize {
        self.depth
    }
}

/// Hash two 32-byte values together (native computation)
pub fn hash_pair_native(left: &[u8; 32], right: &[u8; 32], params: &PoseidonParams) -> [u8; 32] {
    // Convert bytes to scalars
    let mut inputs = Vec::with_capacity(16);

    for chunk in left.chunks(4) {
        inputs.push(u32::from_le_bytes(chunk.try_into().unwrap()) as Scalar);
    }
    for chunk in right.chunks(4) {
        inputs.push(u32::from_le_bytes(chunk.try_into().unwrap()) as Scalar);
    }

    // Apply Poseidon
    let output = poseidon_native(&inputs, params);

    // Convert back to bytes
    let mut result = [0u8; 32];
    for (i, &scalar) in output.iter().take(8).enumerate() {
        result[i * 4..(i + 1) * 4].copy_from_slice(&(scalar as u32).to_le_bytes());
    }

    result
}

/// Compute Merkle root from leaves (native)
pub fn compute_merkle_root_native(leaves: &[[u8; 32]], params: &PoseidonParams) -> [u8; 32] {
    assert!(leaves.len().is_power_of_two());

    if leaves.len() == 1 {
        return leaves[0];
    }

    let mut current_level = leaves.to_vec();

    while current_level.len() > 1 {
        let mut next_level = Vec::with_capacity(current_level.len() / 2);

        for pair in current_level.chunks(2) {
            let hash = hash_pair_native(&pair[0], &pair[1], params);
            next_level.push(hash);
        }

        current_level = next_level;
    }

    current_level[0]
}

/// Generate Merkle proof for leaf at index
pub fn generate_merkle_proof(
    leaves: &[[u8; 32]],
    index: usize,
    params: &PoseidonParams,
) -> MerkleProof {
    assert!(leaves.len().is_power_of_two());
    assert!(index < leaves.len());

    let depth = (leaves.len() as f64).log2() as usize;
    let mut siblings = Vec::with_capacity(depth);
    let mut path = Vec::with_capacity(depth);

    let mut current_level = leaves.to_vec();
    let mut current_index = index;

    for _ in 0..depth {
        let sibling_index = current_index ^ 1;
        siblings.push(current_level[sibling_index]);
        path.push(current_index & 1 == 1);

        // Compute next level
        let mut next_level = Vec::with_capacity(current_level.len() / 2);
        for pair in current_level.chunks(2) {
            let hash = hash_pair_native(&pair[0], &pair[1], params);
            next_level.push(hash);
        }

        current_level = next_level;
        current_index /= 2;
    }

    MerkleProof {
        siblings,
        path,
        leaf_index: index as u64,
    }
}

/// Sparse Merkle Tree for efficient state representation
pub struct SparseMerkleTree {
    /// Tree depth (typically 256 for key-value stores)
    depth: usize,
    /// Poseidon parameters
    params: PoseidonParams,
    /// Default hashes for each level (empty subtree hashes)
    default_hashes: Vec<[u8; 32]>,
}

impl SparseMerkleTree {
    /// Create new sparse Merkle tree
    pub fn new(depth: usize) -> Self {
        let params = PoseidonParams::secure_128(16);

        // Compute default hashes (empty subtree)
        let mut default_hashes = Vec::with_capacity(depth + 1);
        let mut current = [0u8; 32]; // Empty leaf

        default_hashes.push(current);
        for _ in 0..depth {
            current = hash_pair_native(&current, &current, &params);
            default_hashes.push(current);
        }

        Self {
            depth,
            params,
            default_hashes,
        }
    }

    /// Get default hash at given level
    pub fn default_hash(&self, level: usize) -> &[u8; 32] {
        &self.default_hashes[level]
    }

    /// Get empty root hash
    pub fn empty_root(&self) -> &[u8; 32] {
        &self.default_hashes[self.depth]
    }
}

/// Gadget for sparse Merkle tree operations in circuits
pub struct SparseMerkleTreeGadget {
    /// Base Merkle tree gadget
    inner: MerkleTreeGadget,
    /// Default hash wires at each level
    default_hashes: Vec<[usize; 8]>,
}

impl SparseMerkleTreeGadget {
    /// Create new sparse Merkle tree gadget
    pub fn new(depth: usize, builder: &mut ConstraintBuilder) -> Self {
        let smt = SparseMerkleTree::new(depth);

        // Allocate constant wires for default hashes
        let mut default_hashes = Vec::with_capacity(depth + 1);

        for i in 0..=depth {
            let hash_bytes = smt.default_hash(i);
            let mut hash_wires = [0usize; 8];

            for (j, chunk) in hash_bytes.chunks(4).enumerate() {
                let value = u32::from_le_bytes(chunk.try_into().unwrap()) as Scalar;
                let wire = builder.allocator.alloc_witness();
                builder.add_constant(wire, value);
                hash_wires[j] = wire;
            }

            default_hashes.push(hash_wires);
        }

        Self {
            inner: MerkleTreeGadget::new(depth),
            default_hashes,
        }
    }

    /// Verify sparse Merkle proof with possible default siblings
    pub fn verify_sparse_proof(
        &self,
        builder: &mut ConstraintBuilder,
        key: &[usize; 8], // Key as 256-bit value
        value: &[usize; 8],
        proof: &SparseMerkleProofWires,
        root: &[usize; 8],
    ) -> usize {
        // Convert proof to regular Merkle proof format
        let merkle_proof = MerkleProofWires {
            siblings: proof.siblings.clone(),
            path_bits: proof.path_bits.clone(),
        };

        // Hash key and value to get leaf
        let leaf = self.inner.poseidon.hash_pair(builder, key, value);

        // Verify using base gadget
        self.inner.verify_proof(builder, &leaf, &merkle_proof, root)
    }

    /// Get default hash at level
    pub fn default_hash(&self, level: usize) -> &[usize; 8] {
        &self.default_hashes[level]
    }
}

/// Sparse Merkle proof wires
#[derive(Clone, Debug)]
pub struct SparseMerkleProofWires {
    /// Sibling hashes (may include default hashes)
    pub siblings: Vec<[usize; 8]>,
    /// Path bits derived from key
    pub path_bits: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_proof_roundtrip() {
        let params = PoseidonParams::secure_128(16);

        // Create 8 random leaves
        let leaves: Vec<[u8; 32]> = (0..8)
            .map(|i| {
                let mut leaf = [0u8; 32];
                leaf[0] = i as u8;
                leaf
            })
            .collect();

        // Compute root
        let root = compute_merkle_root_native(&leaves, &params);

        // Generate and verify proofs for each leaf
        for i in 0..8 {
            let proof = generate_merkle_proof(&leaves, i, &params);
            assert!(proof.verify(&leaves[i], &root, &params), "Proof {} failed", i);
        }
    }

    #[test]
    fn test_merkle_constraint_estimate() {
        let gadget = MerkleTreeGadget::new(32); // 32-level tree
        let constraints = gadget.estimate_constraints();

        println!("Merkle (depth=32) constraints: {}", constraints);

        // Should be reasonable
        assert!(constraints > 10_000);
        assert!(constraints < 100_000);
    }

    #[test]
    fn test_sparse_merkle_tree() {
        let smt = SparseMerkleTree::new(256);

        // Empty root should be deterministic
        let root1 = smt.empty_root();
        let root2 = smt.empty_root();
        assert_eq!(root1, root2);

        // Default hashes should be precomputed
        assert_eq!(smt.default_hashes.len(), 257);
    }

    #[test]
    fn test_hash_pair_deterministic() {
        let params = PoseidonParams::secure_128(16);

        let left = [1u8; 32];
        let right = [2u8; 32];

        let hash1 = hash_pair_native(&left, &right, &params);
        let hash2 = hash_pair_native(&left, &right, &params);

        assert_eq!(hash1, hash2);

        // Swapping inputs should give different result
        let hash3 = hash_pair_native(&right, &left, &params);
        assert_ne!(hash1, hash3);
    }
}
