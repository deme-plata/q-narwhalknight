//! Merkle Tree for Share Verification
//!
//! v2.3.5-beta: Efficient Merkle tree with proof generation and verification.
//!
//! Miners can prove their share was recorded with O(log n) sized proofs.

use blake3::Hasher;
use serde::{Deserialize, Serialize};

/// Merkle tree for share verification
#[derive(Debug, Clone, Default)]
pub struct MerkleTree {
    /// Leaf hashes (bottom level)
    leaves: Vec<[u8; 32]>,

    /// Internal node levels (level 0 = leaves, level n = root)
    levels: Vec<Vec<[u8; 32]>>,

    /// Cached root hash
    root_hash: [u8; 32],
}

impl MerkleTree {
    /// Build Merkle tree from leaf hashes
    pub fn build(leaves: &[[u8; 32]]) -> Self {
        if leaves.is_empty() {
            return Self {
                leaves: Vec::new(),
                levels: Vec::new(),
                root_hash: [0u8; 32],
            };
        }

        let leaves = leaves.to_vec();
        let mut levels = vec![leaves.clone()];

        // Build tree bottom-up
        let mut current_level = leaves.clone();

        while current_level.len() > 1 {
            let mut next_level = Vec::new();

            for i in (0..current_level.len()).step_by(2) {
                let left = current_level[i];
                let right = if i + 1 < current_level.len() {
                    current_level[i + 1]
                } else {
                    // Duplicate last element for odd count
                    current_level[i]
                };

                let parent = Self::hash_pair(&left, &right);
                next_level.push(parent);
            }

            levels.push(next_level.clone());
            current_level = next_level;
        }

        let root_hash = levels.last().map(|l| l[0]).unwrap_or([0u8; 32]);

        Self {
            leaves,
            levels,
            root_hash,
        }
    }

    /// Hash two nodes together
    fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(left);
        hasher.update(right);
        *hasher.finalize().as_bytes()
    }

    /// Get root hash
    pub fn root(&self) -> [u8; 32] {
        self.root_hash
    }

    /// Get number of leaves
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    /// Check if tree is empty
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Generate proof for leaf at index
    pub fn generate_proof(&self, index: usize) -> Option<MerkleProof> {
        if index >= self.leaves.len() {
            return None;
        }

        let mut proof_path = Vec::new();
        let mut current_index = index;

        // Walk up the tree collecting sibling hashes
        for level in &self.levels[..self.levels.len().saturating_sub(1)] {
            let sibling_index = if current_index % 2 == 0 {
                current_index + 1
            } else {
                current_index - 1
            };

            let sibling = if sibling_index < level.len() {
                level[sibling_index]
            } else {
                // Handle odd count - sibling is same as self
                level[current_index]
            };

            let is_left = current_index % 2 == 1;
            proof_path.push(ProofElement { hash: sibling, is_left });

            // Move to parent index
            current_index /= 2;
        }

        Some(MerkleProof {
            leaf_index: index as u64,
            leaf_hash: self.leaves[index],
            proof_path,
            root_hash: self.root_hash,
        })
    }

    /// Verify a proof against this tree's root
    pub fn verify_proof(&self, proof: &MerkleProof) -> bool {
        proof.verify(&self.root_hash)
    }
}

/// Element in a Merkle proof path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofElement {
    /// Sibling hash at this level
    pub hash: [u8; 32],

    /// True if sibling is on the left
    pub is_left: bool,
}

/// Merkle proof for a single leaf
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    /// Index of the leaf in the tree
    pub leaf_index: u64,

    /// Hash of the leaf
    pub leaf_hash: [u8; 32],

    /// Path from leaf to root (list of sibling hashes)
    pub proof_path: Vec<ProofElement>,

    /// Expected root hash
    pub root_hash: [u8; 32],
}

impl MerkleProof {
    /// Verify proof against expected root
    pub fn verify(&self, expected_root: &[u8; 32]) -> bool {
        let mut current_hash = self.leaf_hash;

        for element in &self.proof_path {
            current_hash = if element.is_left {
                MerkleTree::hash_pair(&element.hash, &current_hash)
            } else {
                MerkleTree::hash_pair(&current_hash, &element.hash)
            };
        }

        current_hash == *expected_root
    }

    /// Get proof size in bytes
    pub fn size_bytes(&self) -> usize {
        // leaf_index (8) + leaf_hash (32) + root_hash (32) + path elements
        8 + 32 + 32 + self.proof_path.len() * (32 + 1)
    }

    /// Get tree depth from proof
    pub fn tree_depth(&self) -> usize {
        self.proof_path.len()
    }

    /// Serialize proof to compact bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.size_bytes());

        bytes.extend_from_slice(&self.leaf_index.to_le_bytes());
        bytes.extend_from_slice(&self.leaf_hash);
        bytes.extend_from_slice(&self.root_hash);
        bytes.push(self.proof_path.len() as u8);

        for element in &self.proof_path {
            bytes.extend_from_slice(&element.hash);
            bytes.push(element.is_left as u8);
        }

        bytes
    }

    /// Deserialize proof from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 73 {
            return None;
        }

        let leaf_index = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
        let leaf_hash: [u8; 32] = bytes[8..40].try_into().ok()?;
        let root_hash: [u8; 32] = bytes[40..72].try_into().ok()?;
        let path_len = bytes[72] as usize;

        if bytes.len() < 73 + path_len * 33 {
            return None;
        }

        let mut proof_path = Vec::with_capacity(path_len);
        let mut offset = 73;

        for _ in 0..path_len {
            let hash: [u8; 32] = bytes[offset..offset + 32].try_into().ok()?;
            let is_left = bytes[offset + 32] != 0;
            proof_path.push(ProofElement { hash, is_left });
            offset += 33;
        }

        Some(Self {
            leaf_index,
            leaf_hash,
            proof_path,
            root_hash,
        })
    }
}

/// Compact Merkle root for checkpoints
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MerkleRoot(pub [u8; 32]);

impl MerkleRoot {
    /// Create from raw bytes
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get inner bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl From<[u8; 32]> for MerkleRoot {
    fn from(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl From<MerkleRoot> for [u8; 32] {
    fn from(root: MerkleRoot) -> Self {
        root.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_tree_empty() {
        let tree = MerkleTree::build(&[]);
        assert!(tree.is_empty());
        assert_eq!(tree.root(), [0u8; 32]);
    }

    #[test]
    fn test_merkle_tree_single() {
        let leaves = [[1u8; 32]];
        let tree = MerkleTree::build(&leaves);

        assert_eq!(tree.len(), 1);
        assert_eq!(tree.root(), leaves[0]); // Single leaf is the root
    }

    #[test]
    fn test_merkle_tree_two() {
        let leaves = [[1u8; 32], [2u8; 32]];
        let tree = MerkleTree::build(&leaves);

        assert_eq!(tree.len(), 2);

        // Verify proof for first leaf
        let proof = tree.generate_proof(0).unwrap();
        assert!(proof.verify(&tree.root()));

        // Verify proof for second leaf
        let proof = tree.generate_proof(1).unwrap();
        assert!(proof.verify(&tree.root()));
    }

    #[test]
    fn test_merkle_tree_many() {
        let leaves: Vec<[u8; 32]> = (0..16)
            .map(|i| {
                let mut arr = [0u8; 32];
                arr[0] = i as u8;
                arr
            })
            .collect();

        let tree = MerkleTree::build(&leaves);

        assert_eq!(tree.len(), 16);

        // Verify all proofs
        for i in 0..16 {
            let proof = tree.generate_proof(i).unwrap();
            assert!(proof.verify(&tree.root()), "Proof {} failed", i);
        }
    }

    #[test]
    fn test_merkle_tree_odd_count() {
        let leaves: Vec<[u8; 32]> = (0..7)
            .map(|i| {
                let mut arr = [0u8; 32];
                arr[0] = i as u8;
                arr
            })
            .collect();

        let tree = MerkleTree::build(&leaves);

        assert_eq!(tree.len(), 7);

        // Verify all proofs
        for i in 0..7 {
            let proof = tree.generate_proof(i).unwrap();
            assert!(proof.verify(&tree.root()), "Proof {} failed", i);
        }
    }

    #[test]
    fn test_proof_serialization() {
        let leaves = [[1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32]];
        let tree = MerkleTree::build(&leaves);

        let proof = tree.generate_proof(2).unwrap();
        let bytes = proof.to_bytes();
        let restored = MerkleProof::from_bytes(&bytes).unwrap();

        assert_eq!(restored.leaf_index, proof.leaf_index);
        assert_eq!(restored.leaf_hash, proof.leaf_hash);
        assert_eq!(restored.root_hash, proof.root_hash);
        assert!(restored.verify(&tree.root()));
    }

    #[test]
    fn test_invalid_proof_fails() {
        let leaves = [[1u8; 32], [2u8; 32]];
        let tree = MerkleTree::build(&leaves);

        let mut proof = tree.generate_proof(0).unwrap();

        // Tamper with proof
        proof.leaf_hash[0] ^= 0xFF;

        assert!(!proof.verify(&tree.root()));
    }
}
