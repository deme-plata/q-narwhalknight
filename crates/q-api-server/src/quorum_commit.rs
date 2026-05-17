/// Validator quorum commit — Phase 1 multi-validator balance agreement
///
/// After applying each block at tip, every validator signs
///   BLAKE3("validator_commit_v1" || height_BE8 || balance_root || prev_block_hash)
/// and broadcasts it to the `/quorum-commit` gossipsub topic.
///
/// When 3-of-4 validators agree on the same (height, balance_root), the block
/// transitions from "accepted by one node" to "agreed by quorum". This is the first
/// step toward trustless block finality without trusting a single node.
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const DOMAIN: &[u8] = b"validator_commit_v1_";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorCommit {
    pub height: u64,
    pub balance_root: [u8; 32],
    pub prev_block_hash: [u8; 32],
    /// Ed25519 verifying key of the signing validator (32 bytes)
    pub verifying_key: [u8; 32],
    /// Ed25519 signature (64 bytes) — stored as Vec<u8> for serde compatibility
    pub signature: Vec<u8>,
}

impl ValidatorCommit {
    pub fn sign(
        height: u64,
        balance_root: [u8; 32],
        prev_block_hash: [u8; 32],
        signing_key: &SigningKey,
    ) -> Self {
        let msg = signing_message(height, &balance_root, &prev_block_hash);
        let sig: Signature = signing_key.sign(&msg);
        Self {
            height,
            balance_root,
            prev_block_hash,
            verifying_key: signing_key.verifying_key().to_bytes(),
            signature: sig.to_bytes().to_vec(),
        }
    }

    pub fn verify(&self) -> bool {
        let Ok(vk) = VerifyingKey::from_bytes(&self.verifying_key) else {
            return false;
        };
        let Ok(sig_bytes): Result<[u8; 64], _> = self.signature.as_slice().try_into() else {
            return false;
        };
        let sig = Signature::from_bytes(&sig_bytes);
        let msg = signing_message(self.height, &self.balance_root, &self.prev_block_hash);
        vk.verify(&msg, &sig).is_ok()
    }
}

fn signing_message(height: u64, balance_root: &[u8; 32], prev_hash: &[u8; 32]) -> Vec<u8> {
    let mut msg = Vec::with_capacity(DOMAIN.len() + 8 + 32 + 32);
    msg.extend_from_slice(DOMAIN);
    msg.extend_from_slice(&height.to_be_bytes());
    msg.extend_from_slice(balance_root);
    msg.extend_from_slice(prev_hash);
    msg
}

/// Collects ValidatorCommit messages and detects when a height reaches quorum.
pub struct QuorumCommitCollector {
    /// height → verified commits (deduplicated by verifying_key)
    commits: RwLock<HashMap<u64, Vec<ValidatorCommit>>>,
    /// minimum number of distinct validators needed to declare quorum
    quorum: usize,
}

impl QuorumCommitCollector {
    pub fn new(quorum: usize) -> Self {
        Self {
            commits: RwLock::new(HashMap::new()),
            quorum,
        }
    }

    /// Add a commit. Returns `Some((count, balance_root))` the first time `quorum`
    /// validators agree on the same (height, balance_root). Returns `None` otherwise.
    pub fn add_commit(&self, commit: ValidatorCommit) -> Option<(usize, [u8; 32])> {
        if !commit.verify() {
            return None;
        }
        let height = commit.height;
        let root = commit.balance_root;
        let mut map = self.commits.write();
        let entries = map.entry(height).or_default();

        // Deduplicate by verifying key — one vote per validator
        if entries.iter().any(|c| c.verifying_key == commit.verifying_key) {
            return None;
        }
        entries.push(commit);

        // Count validators that agree on this root
        let count = entries.iter().filter(|c| c.balance_root == root).count();
        if count == self.quorum {
            Some((count, root))
        } else {
            None
        }
    }

    /// How many distinct validators have committed for a height
    pub fn commit_count(&self, height: u64) -> usize {
        self.commits.read().get(&height).map_or(0, |v| v.len())
    }

    /// Drop commits for heights older than `retain_from` to prevent unbounded growth
    pub fn prune_below(&self, retain_from: u64) {
        self.commits.write().retain(|h, _| *h >= retain_from);
    }
}
