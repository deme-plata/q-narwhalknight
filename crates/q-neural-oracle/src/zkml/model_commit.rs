//! Model Commitment Scheme
//!
//! Provides cryptographic commitments to neural network models
//! for verifiable ML inference.

use super::{ZkMLError, ZkNeuralNetwork, ZkSecurityLevel};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use tracing::{debug, info};

/// Model commitment registry
pub struct ModelRegistry {
    /// Registered models by hash
    models: HashMap<[u8; 32], RegisteredModel>,

    /// Security level
    security_level: ZkSecurityLevel,
}

/// Registered model with metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegisteredModel {
    /// Model commitment
    pub commitment: ModelCommitmentFull,

    /// Registration timestamp
    pub registered_at: u64,

    /// Model version
    pub version: String,

    /// Owner (public key or address)
    pub owner: String,

    /// Description
    pub description: String,

    /// Performance metrics
    pub metrics: Option<ModelMetrics>,
}

/// Full model commitment with Merkle proof support
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelCommitmentFull {
    /// Root hash
    pub root_hash: [u8; 32],

    /// Layer hashes
    pub layer_hashes: Vec<[u8; 32]>,

    /// Parameter count
    pub param_count: usize,

    /// Layer count
    pub layer_count: usize,

    /// Architecture hash
    pub architecture_hash: [u8; 32],

    /// Weights hash (separate from architecture)
    pub weights_hash: [u8; 32],
}

/// Model performance metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Accuracy on test set
    pub accuracy: f64,

    /// Average inference time (ms)
    pub avg_inference_time_ms: f64,

    /// Total inferences verified
    pub verified_inferences: u64,

    /// Proof generation time (ms)
    pub avg_proof_time_ms: f64,
}

impl ModelRegistry {
    /// Create new registry
    pub fn new(security_level: ZkSecurityLevel) -> Self {
        Self {
            models: HashMap::new(),
            security_level,
        }
    }

    /// Register a model
    pub fn register(
        &mut self,
        model: &ZkNeuralNetwork,
        owner: String,
        version: String,
        description: String,
    ) -> Result<[u8; 32], ZkMLError> {
        let commitment = self.compute_full_commitment(model)?;
        let hash = commitment.root_hash;

        if self.models.contains_key(&hash) {
            debug!("Model already registered: {:?}", hash);
            return Ok(hash);
        }

        let registered = RegisteredModel {
            commitment,
            registered_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            version,
            owner,
            description,
            metrics: None,
        };

        self.models.insert(hash, registered);
        info!("📝 Model registered: {:?}", hash);

        Ok(hash)
    }

    /// Compute full commitment for model
    fn compute_full_commitment(
        &self,
        model: &ZkNeuralNetwork,
    ) -> Result<ModelCommitmentFull, ZkMLError> {
        // Compute layer hashes
        let mut layer_hashes = Vec::with_capacity(model.layers.len());

        for layer in &model.layers {
            let layer_bytes = bincode::serialize(layer)
                .map_err(|e| ZkMLError::SerializationError(e.to_string()))?;

            let mut hasher = Sha3_256::new();
            hasher.update(&layer_bytes);
            let hash: [u8; 32] = hasher.finalize().into();
            layer_hashes.push(hash);
        }

        // Compute architecture hash (layer types only)
        let mut arch_hasher = Sha3_256::new();
        for layer in &model.layers {
            let layer_type = match layer {
                super::ZkLayer::Dense { in_features, out_features, .. } => {
                    format!("Dense:{}x{}", in_features, out_features)
                }
                super::ZkLayer::ReLU => "ReLU".to_string(),
                super::ZkLayer::Sigmoid => "Sigmoid".to_string(),
                super::ZkLayer::Softmax => "Softmax".to_string(),
                super::ZkLayer::BatchNorm { .. } => "BatchNorm".to_string(),
            };
            arch_hasher.update(layer_type.as_bytes());
        }
        let architecture_hash: [u8; 32] = arch_hasher.finalize().into();

        // Compute weights hash
        let mut weights_hasher = Sha3_256::new();
        for layer in &model.layers {
            if let super::ZkLayer::Dense { weights, bias, .. } = layer {
                for w in weights {
                    weights_hasher.update(&w.to_le_bytes());
                }
                for b in bias {
                    weights_hasher.update(&b.to_le_bytes());
                }
            }
        }
        let weights_hash: [u8; 32] = weights_hasher.finalize().into();

        // Compute Merkle root from layer hashes
        let root_hash = self.compute_merkle_root(&layer_hashes);

        Ok(ModelCommitmentFull {
            root_hash,
            layer_hashes,
            param_count: model.param_count,
            layer_count: model.layers.len(),
            architecture_hash,
            weights_hash,
        })
    }

    /// Compute Merkle root
    fn compute_merkle_root(&self, hashes: &[[u8; 32]]) -> [u8; 32] {
        if hashes.is_empty() {
            return [0u8; 32];
        }

        if hashes.len() == 1 {
            return hashes[0];
        }

        // Pad to power of 2
        let mut current_level: Vec<[u8; 32]> = hashes.to_vec();
        while current_level.len() & (current_level.len() - 1) != 0 {
            current_level.push(*current_level.last().unwrap());
        }

        // Build tree bottom-up
        while current_level.len() > 1 {
            let mut next_level = Vec::with_capacity(current_level.len() / 2);

            for i in (0..current_level.len()).step_by(2) {
                let mut hasher = Sha3_256::new();
                hasher.update(&current_level[i]);
                hasher.update(&current_level[i + 1]);
                next_level.push(hasher.finalize().into());
            }

            current_level = next_level;
        }

        current_level[0]
    }

    /// Get model by hash
    pub fn get(&self, hash: &[u8; 32]) -> Option<&RegisteredModel> {
        self.models.get(hash)
    }

    /// Verify model matches commitment
    pub fn verify_model(
        &self,
        model: &ZkNeuralNetwork,
        expected_hash: &[u8; 32],
    ) -> Result<bool, ZkMLError> {
        let commitment = self.compute_full_commitment(model)?;
        Ok(&commitment.root_hash == expected_hash)
    }

    /// Generate Merkle proof for specific layer
    pub fn generate_layer_proof(
        &self,
        hash: &[u8; 32],
        layer_index: usize,
    ) -> Option<MerkleProof> {
        let registered = self.models.get(hash)?;

        if layer_index >= registered.commitment.layer_hashes.len() {
            return None;
        }

        // Build Merkle proof path
        let mut proof_path = Vec::new();
        let mut current_index = layer_index;
        let mut hashes = registered.commitment.layer_hashes.clone();

        // Pad to power of 2
        while hashes.len() & (hashes.len() - 1) != 0 {
            hashes.push(*hashes.last().unwrap());
        }

        while hashes.len() > 1 {
            let sibling_index = if current_index % 2 == 0 {
                current_index + 1
            } else {
                current_index - 1
            };

            if sibling_index < hashes.len() {
                proof_path.push(ProofNode {
                    hash: hashes[sibling_index],
                    is_left: current_index % 2 == 1,
                });
            }

            // Move to next level
            let mut next_level = Vec::with_capacity(hashes.len() / 2);
            for i in (0..hashes.len()).step_by(2) {
                let mut hasher = Sha3_256::new();
                hasher.update(&hashes[i]);
                hasher.update(&hashes[i + 1]);
                next_level.push(hasher.finalize().into());
            }

            current_index /= 2;
            hashes = next_level;
        }

        Some(MerkleProof {
            layer_index,
            layer_hash: registered.commitment.layer_hashes[layer_index],
            path: proof_path,
            root: registered.commitment.root_hash,
        })
    }

    /// Verify Merkle proof
    pub fn verify_merkle_proof(&self, proof: &MerkleProof) -> bool {
        let mut current_hash = proof.layer_hash;

        for node in &proof.path {
            let mut hasher = Sha3_256::new();
            if node.is_left {
                hasher.update(&node.hash);
                hasher.update(&current_hash);
            } else {
                hasher.update(&current_hash);
                hasher.update(&node.hash);
            }
            current_hash = hasher.finalize().into();
        }

        current_hash == proof.root
    }

    /// Update model metrics
    pub fn update_metrics(&mut self, hash: &[u8; 32], metrics: ModelMetrics) {
        if let Some(model) = self.models.get_mut(hash) {
            model.metrics = Some(metrics);
        }
    }

    /// List all registered models
    pub fn list_models(&self) -> Vec<([u8; 32], &RegisteredModel)> {
        self.models.iter().map(|(k, v)| (*k, v)).collect()
    }
}

/// Merkle proof for a layer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerkleProof {
    /// Layer index
    pub layer_index: usize,

    /// Layer hash
    pub layer_hash: [u8; 32],

    /// Proof path
    pub path: Vec<ProofNode>,

    /// Root hash
    pub root: [u8; 32],
}

/// Node in Merkle proof path
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofNode {
    /// Hash of sibling
    pub hash: [u8; 32],

    /// Whether sibling is on the left
    pub is_left: bool,
}

/// Incremental model update
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelUpdate {
    /// Original model hash
    pub original_hash: [u8; 32],

    /// Updated layer indices
    pub updated_layers: Vec<usize>,

    /// New layer hashes
    pub new_layer_hashes: Vec<[u8; 32]>,

    /// New root hash
    pub new_root_hash: [u8; 32],

    /// Proof of valid update
    pub update_proof: UpdateProof,
}

/// Proof that update is valid
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateProof {
    /// Merkle proofs for updated layers
    pub layer_proofs: Vec<MerkleProof>,

    /// Signature from owner
    pub owner_signature: Vec<u8>,

    /// Timestamp
    pub timestamp: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkml::FloatLayer;

    #[test]
    fn test_model_registration() {
        let layers = vec![
            FloatLayer::Dense {
                weights: vec![1.0, 0.0, 0.0, 1.0],
                bias: vec![0.0, 0.0],
                in_features: 2,
                out_features: 2,
            },
            FloatLayer::ReLU,
        ];

        let model = ZkNeuralNetwork::from_float_model(layers, 16).unwrap();

        let mut registry = ModelRegistry::new(ZkSecurityLevel::PQ128);

        let hash = registry
            .register(
                &model,
                "owner123".to_string(),
                "1.0.0".to_string(),
                "Test model".to_string(),
            )
            .unwrap();

        let registered = registry.get(&hash).unwrap();
        assert_eq!(registered.version, "1.0.0");
        assert_eq!(registered.commitment.layer_count, 2);
    }

    #[test]
    fn test_merkle_proof() {
        let layers = vec![
            FloatLayer::Dense {
                weights: vec![1.0; 4],
                bias: vec![0.0, 0.0],
                in_features: 2,
                out_features: 2,
            },
            FloatLayer::ReLU,
            FloatLayer::Dense {
                weights: vec![1.0; 4],
                bias: vec![0.0, 0.0],
                in_features: 2,
                out_features: 2,
            },
            FloatLayer::ReLU,
        ];

        let model = ZkNeuralNetwork::from_float_model(layers, 16).unwrap();

        let mut registry = ModelRegistry::new(ZkSecurityLevel::PQ128);
        let hash = registry
            .register(&model, "owner".to_string(), "1.0".to_string(), "".to_string())
            .unwrap();

        // Generate and verify proof for layer 1
        let proof = registry.generate_layer_proof(&hash, 1).unwrap();
        assert!(registry.verify_merkle_proof(&proof));

        // Generate proof for layer 3
        let proof3 = registry.generate_layer_proof(&hash, 3).unwrap();
        assert!(registry.verify_merkle_proof(&proof3));
    }

    #[test]
    fn test_model_verification() {
        let layers = vec![FloatLayer::Dense {
            weights: vec![1.0, 2.0, 3.0, 4.0],
            bias: vec![0.1, 0.2],
            in_features: 2,
            out_features: 2,
        }];

        let model = ZkNeuralNetwork::from_float_model(layers, 16).unwrap();

        let mut registry = ModelRegistry::new(ZkSecurityLevel::PQ128);
        let hash = registry
            .register(&model, "owner".to_string(), "1.0".to_string(), "".to_string())
            .unwrap();

        assert!(registry.verify_model(&model, &hash).unwrap());

        // Different model should not match
        let other_layers = vec![FloatLayer::Dense {
            weights: vec![5.0, 6.0, 7.0, 8.0], // Different weights
            bias: vec![0.1, 0.2],
            in_features: 2,
            out_features: 2,
        }];
        let other_model = ZkNeuralNetwork::from_float_model(other_layers, 16).unwrap();
        assert!(!registry.verify_model(&other_model, &hash).unwrap());
    }
}
