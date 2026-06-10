//! Sparse Merkle Trie (SMT) Implementation for State Root Verification
//!
//! This module provides a cryptographically secure sparse Merkle trie optimized for
//! blockchain state management. Key features:
//!
//! - **256-bit key space**: Supports any key that can be hashed to 32 bytes
//! - **Compact proofs**: O(log n) proof size where n is key space (256 bits = 32 levels)
//! - **RocksDB persistence**: Nodes are persisted to a dedicated column family
//! - **Lazy pruning**: Old roots are retained for historical state queries
//! - **Blake3 hashing**: Fast cryptographic hashing (3GB/s on modern CPUs)
//!
//! # Architecture
//!
//! ```text
//! Root [32 bytes]
//!   ├── Internal Node (if bit=0, go left; if bit=1, go right)
//!   │     ├── Left child hash [32 bytes]
//!   │     └── Right child hash [32 bytes]
//!   └── Leaf Node
//!         ├── Key hash [32 bytes]
//!         └── Value hash [32 bytes]
//! ```
//!
//! # Security Properties
//!
//! - **Collision resistance**: Blake3 provides 128-bit collision resistance
//! - **Preimage resistance**: Cannot derive key/value from hash
//! - **Second preimage resistance**: Cannot find alternative key/value with same hash
//! - **Non-membership proofs**: Can prove a key does NOT exist in the trie
//!
//! # Usage
//!
//! ```ignore
//! let smt = SparseMerkleTrie::new(db, "CF_STATE_TRIE");
//!
//! // Insert key-value pairs
//! smt.insert(&key, &value)?;
//!
//! // Get current state root
//! let root = smt.root();
//!
//! // Generate inclusion proof
//! let proof = smt.prove(&key)?;
//!
//! // Verify proof (can be done without access to full trie)
//! assert!(SparseMerkleTrie::verify_proof(&root, &key, Some(&value), &proof));
//! ```

use blake3::Hasher;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{debug, trace, warn};

/// Column family name for trie nodes
pub const CF_STATE_TRIE: &str = "state_trie";

/// Empty hash constant (hash of empty node)
/// This is Blake3("")
pub const EMPTY_HASH: [u8; 32] = [
    0xaf, 0x13, 0x49, 0xb9, 0xf5, 0xf9, 0xa1, 0xa6,
    0xa0, 0x40, 0x4d, 0xea, 0x36, 0xdc, 0xc9, 0x49,
    0x9b, 0xcb, 0x25, 0xc9, 0xad, 0xc1, 0x12, 0xb7,
    0xcc, 0x9a, 0x93, 0xca, 0xe4, 0x1f, 0x32, 0x62,
];

/// Node types in the sparse Merkle trie
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrieNode {
    /// Empty node (represents absence of data)
    Empty,
    /// Leaf node containing key hash and value hash
    Leaf {
        key_hash: [u8; 32],
        value_hash: [u8; 32],
    },
    /// Internal node with left and right child hashes
    Internal {
        left: [u8; 32],
        right: [u8; 32],
    },
}

impl TrieNode {
    /// Serialize node to bytes for storage
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            TrieNode::Empty => vec![0x00],
            TrieNode::Leaf { key_hash, value_hash } => {
                let mut bytes = vec![0x01];
                bytes.extend_from_slice(key_hash);
                bytes.extend_from_slice(value_hash);
                bytes
            }
            TrieNode::Internal { left, right } => {
                let mut bytes = vec![0x02];
                bytes.extend_from_slice(left);
                bytes.extend_from_slice(right);
                bytes
            }
        }
    }

    /// Deserialize node from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.is_empty() {
            return None;
        }
        match bytes[0] {
            0x00 => Some(TrieNode::Empty),
            0x01 if bytes.len() >= 65 => {
                let mut key_hash = [0u8; 32];
                let mut value_hash = [0u8; 32];
                key_hash.copy_from_slice(&bytes[1..33]);
                value_hash.copy_from_slice(&bytes[33..65]);
                Some(TrieNode::Leaf { key_hash, value_hash })
            }
            0x02 if bytes.len() >= 65 => {
                let mut left = [0u8; 32];
                let mut right = [0u8; 32];
                left.copy_from_slice(&bytes[1..33]);
                right.copy_from_slice(&bytes[33..65]);
                Some(TrieNode::Internal { left, right })
            }
            _ => None,
        }
    }

    /// Compute hash of this node
    pub fn hash(&self) -> [u8; 32] {
        match self {
            TrieNode::Empty => EMPTY_HASH,
            TrieNode::Leaf { key_hash, value_hash } => {
                let mut hasher = Hasher::new();
                hasher.update(&[0x01]); // Leaf prefix
                hasher.update(key_hash);
                hasher.update(value_hash);
                *hasher.finalize().as_bytes()
            }
            TrieNode::Internal { left, right } => {
                let mut hasher = Hasher::new();
                hasher.update(&[0x02]); // Internal prefix
                hasher.update(left);
                hasher.update(right);
                *hasher.finalize().as_bytes()
            }
        }
    }
}

/// Merkle proof for inclusion/exclusion verification
#[derive(Debug, Clone)]
pub struct MerkleProof {
    /// Sibling hashes along the path from leaf to root
    /// Index 0 is closest to leaf, index 255 is closest to root
    pub siblings: Vec<[u8; 32]>,
    /// Bitmap indicating which siblings are on the left (0) or right (1)
    pub path_bits: Vec<bool>,
    /// The leaf node at the path (None for non-membership proof)
    pub leaf: Option<TrieNode>,
}

impl MerkleProof {
    /// Serialize proof for transmission
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Number of siblings (max 256)
        bytes.push(self.siblings.len() as u8);

        // Path bits as packed bytes
        let path_bytes: Vec<u8> = self.path_bits
            .chunks(8)
            .map(|chunk| {
                chunk.iter().enumerate().fold(0u8, |acc, (i, &bit)| {
                    if bit { acc | (1 << i) } else { acc }
                })
            })
            .collect();
        bytes.push(path_bytes.len() as u8);
        bytes.extend_from_slice(&path_bytes);

        // Siblings
        for sibling in &self.siblings {
            bytes.extend_from_slice(sibling);
        }

        // Leaf (if present)
        match &self.leaf {
            Some(node) => {
                bytes.push(0x01);
                bytes.extend_from_slice(&node.to_bytes());
            }
            None => {
                bytes.push(0x00);
            }
        }

        bytes
    }

    /// Deserialize proof from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 2 {
            return None;
        }

        let num_siblings = bytes[0] as usize;
        let num_path_bytes = bytes[1] as usize;

        if bytes.len() < 2 + num_path_bytes + num_siblings * 32 + 1 {
            return None;
        }

        // Parse path bits
        let path_bytes = &bytes[2..2 + num_path_bytes];
        let mut path_bits = Vec::with_capacity(num_siblings);
        for (i, &byte) in path_bytes.iter().enumerate() {
            for j in 0..8 {
                if i * 8 + j < num_siblings {
                    path_bits.push((byte >> j) & 1 == 1);
                }
            }
        }

        // Parse siblings
        let mut siblings = Vec::with_capacity(num_siblings);
        let sibling_start = 2 + num_path_bytes;
        for i in 0..num_siblings {
            let mut sibling = [0u8; 32];
            sibling.copy_from_slice(&bytes[sibling_start + i * 32..sibling_start + (i + 1) * 32]);
            siblings.push(sibling);
        }

        // Parse leaf
        let leaf_marker_pos = sibling_start + num_siblings * 32;
        let leaf = if bytes[leaf_marker_pos] == 0x01 {
            TrieNode::from_bytes(&bytes[leaf_marker_pos + 1..])
        } else {
            None
        };

        Some(MerkleProof {
            siblings,
            path_bits,
            leaf,
        })
    }
}

/// Sparse Merkle Trie with optional RocksDB persistence
pub struct SparseMerkleTrie {
    /// In-memory cache of nodes (hash -> node)
    cache: RwLock<HashMap<[u8; 32], TrieNode>>,
    /// Current root hash
    root: RwLock<[u8; 32]>,
    /// RocksDB handle for persistence (optional)
    #[cfg(not(target_os = "windows"))]
    db: Option<Arc<rocksdb::DB>>,
    /// Column family name
    cf_name: String,
    /// Number of trie levels (256 for 32-byte keys)
    depth: usize,
    /// v10.0.9: Maximum cache entries before eviction (prevents 20GB+ memory leak during sync)
    /// When exceeded, cache is cleared — nodes reload from RocksDB on demand.
    max_cache_entries: usize,
}

impl SparseMerkleTrie {
    /// Create a new in-memory sparse Merkle trie
    pub fn new_in_memory() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            root: RwLock::new(EMPTY_HASH),
            #[cfg(not(target_os = "windows"))]
            db: None,
            cf_name: CF_STATE_TRIE.to_string(),
            depth: 256,
            max_cache_entries: 500_000, // ~70MB — balances sync speed vs memory (evicts every ~27K blocks)
        }
    }

    /// Create a sparse Merkle trie backed by RocksDB
    #[cfg(not(target_os = "windows"))]
    pub fn new_with_db(db: Arc<rocksdb::DB>, cf_name: &str) -> Self {
        // Try to load existing root from DB
        let root = Self::load_root_from_db(&db, cf_name).unwrap_or(EMPTY_HASH);

        Self {
            cache: RwLock::new(HashMap::new()),
            root: RwLock::new(root),
            db: Some(db),
            cf_name: cf_name.to_string(),
            depth: 256,
            max_cache_entries: 500_000, // ~70MB — balances sync speed vs memory (evicts every ~27K blocks)
        }
    }

    /// Load root hash from database
    #[cfg(not(target_os = "windows"))]
    fn load_root_from_db(db: &rocksdb::DB, cf_name: &str) -> Option<[u8; 32]> {
        let cf = db.cf_handle(cf_name)?;
        let root_key = b"__ROOT__";
        let bytes = db.get_cf(&cf, root_key).ok()??;
        if bytes.len() == 32 {
            let mut root = [0u8; 32];
            root.copy_from_slice(&bytes);
            Some(root)
        } else {
            None
        }
    }

    /// Get current state root
    pub fn root(&self) -> [u8; 32] {
        *self.root.read().unwrap()
    }

    /// Check if trie is empty
    pub fn is_empty(&self) -> bool {
        self.root() == EMPTY_HASH
    }

    /// Hash a key to get the path in the trie
    pub fn hash_key(key: &[u8]) -> [u8; 32] {
        *blake3::hash(key).as_bytes()
    }

    /// Hash a value
    pub fn hash_value(value: &[u8]) -> [u8; 32] {
        *blake3::hash(value).as_bytes()
    }

    /// Get bit at position in hash (0 = leftmost bit)
    fn get_bit(hash: &[u8; 32], position: usize) -> bool {
        let byte_index = position / 8;
        let bit_index = 7 - (position % 8); // MSB first
        (hash[byte_index] >> bit_index) & 1 == 1
    }

    /// Get a node by its hash
    fn get_node(&self, hash: &[u8; 32]) -> Option<TrieNode> {
        // Check if it's the empty hash
        if hash == &EMPTY_HASH {
            return Some(TrieNode::Empty);
        }

        // Check cache first
        {
            let cache = self.cache.read().unwrap();
            if let Some(node) = cache.get(hash) {
                return Some(node.clone());
            }
        }

        // Try loading from DB
        #[cfg(not(target_os = "windows"))]
        if let Some(ref db) = self.db {
            if let Some(cf) = db.cf_handle(&self.cf_name) {
                if let Ok(Some(bytes)) = db.get_cf(&cf, hash) {
                    if let Some(node) = TrieNode::from_bytes(&bytes) {
                        // Cache the loaded node
                        let mut cache = self.cache.write().unwrap();
                        cache.insert(*hash, node.clone());
                        return Some(node);
                    }
                }
            }
        }

        None
    }

    /// Store a node and return its hash
    fn put_node(&self, node: &TrieNode) -> [u8; 32] {
        let hash = node.hash();

        // Don't store empty nodes
        if matches!(node, TrieNode::Empty) {
            return EMPTY_HASH;
        }

        // Store in cache, with eviction when over limit
        {
            let mut cache = self.cache.write().unwrap();
            // v10.0.9: Evict cache when it grows too large to prevent 20GB+ memory leak.
            // During sync, millions of trie operations create orphaned nodes that accumulate.
            // Nodes are persisted in RocksDB, so clearing the cache only costs a disk read on miss.
            if cache.len() >= self.max_cache_entries {
                warn!("🧹 [SMT] Cache eviction: {} entries exceeded {} limit, clearing",
                      cache.len(), self.max_cache_entries);
                cache.clear();
            }
            cache.insert(hash, node.clone());
        }

        // Persist to DB
        #[cfg(not(target_os = "windows"))]
        if let Some(ref db) = self.db {
            if let Some(cf) = db.cf_handle(&self.cf_name) {
                let bytes = node.to_bytes();
                if let Err(e) = db.put_cf(&cf, &hash, &bytes) {
                    warn!("Failed to persist trie node: {}", e);
                }
            }
        }

        hash
    }

    /// Insert a key-value pair into the trie
    pub fn insert(&self, key: &[u8], value: &[u8]) -> [u8; 32] {
        let key_hash = Self::hash_key(key);
        let value_hash = Self::hash_value(value);

        trace!("SMT insert: key_hash={} value_hash={}",
               hex::encode(&key_hash[..8]), hex::encode(&value_hash[..8]));

        let current_root = self.root();
        let new_root = self.insert_recursive(&current_root, &key_hash, &value_hash, 0);

        // Update root
        {
            let mut root = self.root.write().unwrap();
            *root = new_root;
        }

        // Persist root to DB
        #[cfg(not(target_os = "windows"))]
        if let Some(ref db) = self.db {
            if let Some(cf) = db.cf_handle(&self.cf_name) {
                let root_key = b"__ROOT__";
                if let Err(e) = db.put_cf(&cf, root_key, &new_root) {
                    warn!("Failed to persist root: {}", e);
                }
            }
        }

        new_root
    }

    /// Recursive insert helper
    fn insert_recursive(
        &self,
        node_hash: &[u8; 32],
        key_hash: &[u8; 32],
        value_hash: &[u8; 32],
        depth: usize,
    ) -> [u8; 32] {
        // Reached max depth - create leaf
        if depth >= self.depth {
            let leaf = TrieNode::Leaf {
                key_hash: *key_hash,
                value_hash: *value_hash,
            };
            return self.put_node(&leaf);
        }

        let node = self.get_node(node_hash).unwrap_or(TrieNode::Empty);

        match node {
            TrieNode::Empty => {
                // Empty node - create a leaf directly
                let leaf = TrieNode::Leaf {
                    key_hash: *key_hash,
                    value_hash: *value_hash,
                };
                self.put_node(&leaf)
            }
            TrieNode::Leaf { key_hash: existing_key, value_hash: existing_value } => {
                if &existing_key == key_hash {
                    // Same key - update value
                    let leaf = TrieNode::Leaf {
                        key_hash: *key_hash,
                        value_hash: *value_hash,
                    };
                    self.put_node(&leaf)
                } else {
                    // Different key - need to split into internal node
                    // Find first differing bit
                    let existing_bit = Self::get_bit(&existing_key, depth);
                    let new_bit = Self::get_bit(key_hash, depth);

                    if existing_bit == new_bit {
                        // Same direction - recurse deeper
                        let existing_leaf = TrieNode::Leaf {
                            key_hash: existing_key,
                            value_hash: existing_value,
                        };
                        let existing_hash = existing_leaf.hash();

                        let child_hash = self.insert_recursive(&existing_hash, key_hash, value_hash, depth + 1);

                        let internal = if new_bit {
                            TrieNode::Internal { left: EMPTY_HASH, right: child_hash }
                        } else {
                            TrieNode::Internal { left: child_hash, right: EMPTY_HASH }
                        };
                        self.put_node(&internal)
                    } else {
                        // Different direction - create internal with both leaves
                        let existing_leaf = TrieNode::Leaf {
                            key_hash: existing_key,
                            value_hash: existing_value,
                        };
                        let existing_hash = self.put_node(&existing_leaf);

                        let new_leaf = TrieNode::Leaf {
                            key_hash: *key_hash,
                            value_hash: *value_hash,
                        };
                        let new_hash = self.put_node(&new_leaf);

                        let internal = if new_bit {
                            TrieNode::Internal { left: existing_hash, right: new_hash }
                        } else {
                            TrieNode::Internal { left: new_hash, right: existing_hash }
                        };
                        self.put_node(&internal)
                    }
                }
            }
            TrieNode::Internal { left, right } => {
                let bit = Self::get_bit(key_hash, depth);
                if bit {
                    // Go right
                    let new_right = self.insert_recursive(&right, key_hash, value_hash, depth + 1);
                    let internal = TrieNode::Internal { left, right: new_right };
                    self.put_node(&internal)
                } else {
                    // Go left
                    let new_left = self.insert_recursive(&left, key_hash, value_hash, depth + 1);
                    let internal = TrieNode::Internal { left: new_left, right };
                    self.put_node(&internal)
                }
            }
        }
    }

    /// Delete a key from the trie
    pub fn delete(&self, key: &[u8]) -> [u8; 32] {
        let key_hash = Self::hash_key(key);
        let current_root = self.root();
        let new_root = self.delete_recursive(&current_root, &key_hash, 0);

        // Update root
        {
            let mut root = self.root.write().unwrap();
            *root = new_root;
        }

        // Persist root to DB
        #[cfg(not(target_os = "windows"))]
        if let Some(ref db) = self.db {
            if let Some(cf) = db.cf_handle(&self.cf_name) {
                let root_key = b"__ROOT__";
                if let Err(e) = db.put_cf(&cf, root_key, &new_root) {
                    warn!("Failed to persist root after delete: {}", e);
                }
            }
        }

        new_root
    }

    /// Recursive delete helper
    fn delete_recursive(&self, node_hash: &[u8; 32], key_hash: &[u8; 32], depth: usize) -> [u8; 32] {
        if depth >= self.depth {
            return EMPTY_HASH;
        }

        let node = match self.get_node(node_hash) {
            Some(n) => n,
            None => return EMPTY_HASH,
        };

        match node {
            TrieNode::Empty => EMPTY_HASH,
            TrieNode::Leaf { key_hash: existing_key, .. } => {
                if &existing_key == key_hash {
                    EMPTY_HASH
                } else {
                    *node_hash // Keep existing leaf
                }
            }
            TrieNode::Internal { left, right } => {
                let bit = Self::get_bit(key_hash, depth);
                let (new_left, new_right) = if bit {
                    let new_right = self.delete_recursive(&right, key_hash, depth + 1);
                    (left, new_right)
                } else {
                    let new_left = self.delete_recursive(&left, key_hash, depth + 1);
                    (new_left, right)
                };

                // Collapse if both children are empty or only one leaf remains
                if new_left == EMPTY_HASH && new_right == EMPTY_HASH {
                    EMPTY_HASH
                } else if new_left == EMPTY_HASH {
                    // Only right child - check if it's a leaf we can promote
                    if let Some(TrieNode::Leaf { .. }) = self.get_node(&new_right) {
                        new_right
                    } else {
                        let internal = TrieNode::Internal { left: new_left, right: new_right };
                        self.put_node(&internal)
                    }
                } else if new_right == EMPTY_HASH {
                    // Only left child - check if it's a leaf we can promote
                    if let Some(TrieNode::Leaf { .. }) = self.get_node(&new_left) {
                        new_left
                    } else {
                        let internal = TrieNode::Internal { left: new_left, right: new_right };
                        self.put_node(&internal)
                    }
                } else {
                    let internal = TrieNode::Internal { left: new_left, right: new_right };
                    self.put_node(&internal)
                }
            }
        }
    }

    /// Get value for a key (returns value hash if exists)
    pub fn get(&self, key: &[u8]) -> Option<[u8; 32]> {
        let key_hash = Self::hash_key(key);
        self.get_by_hash(&key_hash)
    }

    /// Get value by key hash
    pub fn get_by_hash(&self, key_hash: &[u8; 32]) -> Option<[u8; 32]> {
        let current_root = self.root();
        self.get_recursive(&current_root, key_hash, 0)
    }

    /// Recursive get helper
    fn get_recursive(&self, node_hash: &[u8; 32], key_hash: &[u8; 32], depth: usize) -> Option<[u8; 32]> {
        if depth >= self.depth {
            return None;
        }

        let node = self.get_node(node_hash)?;

        match node {
            TrieNode::Empty => None,
            TrieNode::Leaf { key_hash: existing_key, value_hash } => {
                if &existing_key == key_hash {
                    Some(value_hash)
                } else {
                    None
                }
            }
            TrieNode::Internal { left, right } => {
                let bit = Self::get_bit(key_hash, depth);
                if bit {
                    self.get_recursive(&right, key_hash, depth + 1)
                } else {
                    self.get_recursive(&left, key_hash, depth + 1)
                }
            }
        }
    }

    /// Generate a Merkle proof for a key
    pub fn prove(&self, key: &[u8]) -> MerkleProof {
        let key_hash = Self::hash_key(key);
        let current_root = self.root();
        self.prove_recursive(&current_root, &key_hash, 0)
    }

    /// Recursive proof generation
    fn prove_recursive(&self, node_hash: &[u8; 32], key_hash: &[u8; 32], depth: usize) -> MerkleProof {
        if depth >= self.depth {
            return MerkleProof {
                siblings: Vec::new(),
                path_bits: Vec::new(),
                leaf: None,
            };
        }

        let node = self.get_node(node_hash).unwrap_or(TrieNode::Empty);

        match node {
            TrieNode::Empty => MerkleProof {
                siblings: Vec::new(),
                path_bits: Vec::new(),
                leaf: None,
            },
            TrieNode::Leaf { key_hash: existing_key, value_hash } => {
                let leaf = if &existing_key == key_hash {
                    Some(TrieNode::Leaf { key_hash: existing_key, value_hash })
                } else {
                    // Non-membership: return the conflicting leaf
                    Some(TrieNode::Leaf { key_hash: existing_key, value_hash })
                };
                MerkleProof {
                    siblings: Vec::new(),
                    path_bits: Vec::new(),
                    leaf,
                }
            }
            TrieNode::Internal { left, right } => {
                let bit = Self::get_bit(key_hash, depth);
                let (child_hash, sibling_hash) = if bit {
                    (right, left)
                } else {
                    (left, right)
                };

                let mut proof = self.prove_recursive(&child_hash, key_hash, depth + 1);
                proof.siblings.push(sibling_hash);
                proof.path_bits.push(bit);
                proof
            }
        }
    }

    /// Verify a Merkle proof
    ///
    /// Returns true if the proof is valid for the given root, key, and value.
    /// If value is None, this verifies non-membership.
    pub fn verify_proof(
        root: &[u8; 32],
        key: &[u8],
        value: Option<&[u8]>,
        proof: &MerkleProof,
    ) -> bool {
        let key_hash = Self::hash_key(key);
        let value_hash = value.map(Self::hash_value);

        // Start from the leaf
        let mut current_hash = match &proof.leaf {
            Some(TrieNode::Leaf { key_hash: leaf_key, value_hash: leaf_value }) => {
                // Check if this is the key we're looking for
                if let Some(expected_value) = &value_hash {
                    if leaf_key != &key_hash || leaf_value != expected_value {
                        return false; // Key or value mismatch
                    }
                } else {
                    // Non-membership proof: leaf key should be different
                    if leaf_key == &key_hash {
                        return false; // Key exists, but we expected non-membership
                    }
                }
                TrieNode::Leaf {
                    key_hash: *leaf_key,
                    value_hash: *leaf_value,
                }.hash()
            }
            Some(_) => return false, // Invalid leaf type
            None => {
                if value.is_some() {
                    return false; // Expected membership but no leaf
                }
                EMPTY_HASH
            }
        };

        // Walk up the tree
        for (sibling, &bit) in proof.siblings.iter().zip(proof.path_bits.iter()) {
            let (left, right) = if bit {
                (*sibling, current_hash)
            } else {
                (current_hash, *sibling)
            };

            let internal = TrieNode::Internal { left, right };
            current_hash = internal.hash();
        }

        &current_hash == root
    }

    /// Batch insert multiple key-value pairs efficiently
    pub fn batch_insert(&self, entries: &[(&[u8], &[u8])]) -> [u8; 32] {
        for (key, value) in entries {
            self.insert(key, value);
        }
        self.root()
    }

    /// Get statistics about the trie
    pub fn stats(&self) -> TrieStats {
        let cache = self.cache.read().unwrap();
        let mut leaf_count = 0;
        let mut internal_count = 0;

        for node in cache.values() {
            match node {
                TrieNode::Leaf { .. } => leaf_count += 1,
                TrieNode::Internal { .. } => internal_count += 1,
                TrieNode::Empty => {}
            }
        }

        TrieStats {
            root: self.root(),
            cached_nodes: cache.len(),
            leaf_count,
            internal_count,
        }
    }

    /// Clear the in-memory cache (nodes remain in DB)
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }
}

/// Statistics about the trie
#[derive(Debug, Clone)]
pub struct TrieStats {
    pub root: [u8; 32],
    pub cached_nodes: usize,
    pub leaf_count: usize,
    pub internal_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_trie() {
        let smt = SparseMerkleTrie::new_in_memory();
        assert!(smt.is_empty());
        assert_eq!(smt.root(), EMPTY_HASH);
    }

    #[test]
    fn test_single_insert() {
        let smt = SparseMerkleTrie::new_in_memory();
        let key = b"hello";
        let value = b"world";

        let root = smt.insert(key, value);
        assert_ne!(root, EMPTY_HASH);
        assert!(!smt.is_empty());

        // Should be able to retrieve
        let retrieved = smt.get(key);
        assert_eq!(retrieved, Some(SparseMerkleTrie::hash_value(value)));
    }

    #[test]
    fn test_multiple_inserts() {
        let smt = SparseMerkleTrie::new_in_memory();

        smt.insert(b"key1", b"value1");
        smt.insert(b"key2", b"value2");
        smt.insert(b"key3", b"value3");

        assert_eq!(smt.get(b"key1"), Some(SparseMerkleTrie::hash_value(b"value1")));
        assert_eq!(smt.get(b"key2"), Some(SparseMerkleTrie::hash_value(b"value2")));
        assert_eq!(smt.get(b"key3"), Some(SparseMerkleTrie::hash_value(b"value3")));
        assert_eq!(smt.get(b"nonexistent"), None);
    }

    #[test]
    fn test_update_value() {
        let smt = SparseMerkleTrie::new_in_memory();
        let key = b"mykey";

        smt.insert(key, b"value1");
        let root1 = smt.root();

        smt.insert(key, b"value2");
        let root2 = smt.root();

        assert_ne!(root1, root2);
        assert_eq!(smt.get(key), Some(SparseMerkleTrie::hash_value(b"value2")));
    }

    #[test]
    fn test_delete() {
        let smt = SparseMerkleTrie::new_in_memory();

        smt.insert(b"key1", b"value1");
        smt.insert(b"key2", b"value2");

        assert!(smt.get(b"key1").is_some());

        smt.delete(b"key1");
        assert!(smt.get(b"key1").is_none());
        assert!(smt.get(b"key2").is_some());
    }

    #[test]
    fn test_proof_membership() {
        let smt = SparseMerkleTrie::new_in_memory();
        let key = b"testkey";
        let value = b"testvalue";

        smt.insert(key, value);
        let root = smt.root();
        let proof = smt.prove(key);

        assert!(SparseMerkleTrie::verify_proof(&root, key, Some(value), &proof));
    }

    #[test]
    fn test_proof_non_membership() {
        let smt = SparseMerkleTrie::new_in_memory();

        smt.insert(b"key1", b"value1");
        let root = smt.root();
        let proof = smt.prove(b"nonexistent");

        assert!(SparseMerkleTrie::verify_proof(&root, b"nonexistent", None, &proof));
    }

    #[test]
    fn test_proof_invalid() {
        let smt = SparseMerkleTrie::new_in_memory();
        let key = b"testkey";
        let value = b"testvalue";

        smt.insert(key, value);
        let root = smt.root();
        let proof = smt.prove(key);

        // Wrong value should fail
        assert!(!SparseMerkleTrie::verify_proof(&root, key, Some(b"wrongvalue"), &proof));

        // Wrong root should fail
        let wrong_root = [0u8; 32];
        assert!(!SparseMerkleTrie::verify_proof(&wrong_root, key, Some(value), &proof));
    }

    #[test]
    fn test_proof_serialization() {
        let smt = SparseMerkleTrie::new_in_memory();

        smt.insert(b"key1", b"value1");
        smt.insert(b"key2", b"value2");

        let proof = smt.prove(b"key1");
        let bytes = proof.to_bytes();
        let restored = MerkleProof::from_bytes(&bytes).unwrap();

        let root = smt.root();
        assert!(SparseMerkleTrie::verify_proof(&root, b"key1", Some(b"value1"), &restored));
    }

    #[test]
    fn test_deterministic_root() {
        // Same insertions should produce same root
        let smt1 = SparseMerkleTrie::new_in_memory();
        let smt2 = SparseMerkleTrie::new_in_memory();

        for i in 0..10 {
            let key = format!("key{}", i);
            let value = format!("value{}", i);
            smt1.insert(key.as_bytes(), value.as_bytes());
            smt2.insert(key.as_bytes(), value.as_bytes());
        }

        assert_eq!(smt1.root(), smt2.root());
    }

    #[test]
    fn test_batch_insert() {
        let smt = SparseMerkleTrie::new_in_memory();

        let entries: Vec<(&[u8], &[u8])> = vec![
            (b"key1", b"value1"),
            (b"key2", b"value2"),
            (b"key3", b"value3"),
        ];

        let root = smt.batch_insert(&entries);
        assert_ne!(root, EMPTY_HASH);

        assert_eq!(smt.get(b"key1"), Some(SparseMerkleTrie::hash_value(b"value1")));
        assert_eq!(smt.get(b"key2"), Some(SparseMerkleTrie::hash_value(b"value2")));
        assert_eq!(smt.get(b"key3"), Some(SparseMerkleTrie::hash_value(b"value3")));
    }

    #[test]
    fn test_node_serialization() {
        // Empty node
        let empty = TrieNode::Empty;
        assert_eq!(TrieNode::from_bytes(&empty.to_bytes()), Some(TrieNode::Empty));

        // Leaf node
        let leaf = TrieNode::Leaf {
            key_hash: [1u8; 32],
            value_hash: [2u8; 32],
        };
        assert_eq!(TrieNode::from_bytes(&leaf.to_bytes()), Some(leaf.clone()));

        // Internal node
        let internal = TrieNode::Internal {
            left: [3u8; 32],
            right: [4u8; 32],
        };
        assert_eq!(TrieNode::from_bytes(&internal.to_bytes()), Some(internal.clone()));
    }
}
