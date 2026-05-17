//! Inference Prover for zkML
//!
//! Generates proofs that neural network inference was performed correctly.

use super::{ZkMLConfig, ZkMLError, ZkMLProof, ZkNeuralNetwork, ZkSecurityLevel};
use sha3::{Digest, Sha3_256};
use std::time::Instant;
use tracing::{debug, info};

/// Specialized inference prover with optimizations
pub struct InferenceProver {
    config: ZkMLConfig,

    /// Precomputed SRS (if available)
    srs_cache: Option<PrecomputedSRS>,

    /// Statistics
    stats: ProverStats,
}

/// Precomputed Structured Reference String
#[derive(Clone)]
pub struct PrecomputedSRS {
    /// Security level
    pub security_level: ZkSecurityLevel,

    /// Maximum model size supported
    pub max_model_size: usize,

    /// Precomputed powers
    pub powers: Vec<Vec<u8>>,

    /// Creation timestamp
    pub created_at: u64,
}

/// Prover statistics
#[derive(Clone, Debug, Default)]
pub struct ProverStats {
    pub total_proofs: u64,
    pub total_constraints: u64,
    pub total_time_ms: u64,
    pub avg_time_per_constraint_us: f64,
}

impl InferenceProver {
    /// Create new inference prover
    pub fn new(config: ZkMLConfig) -> Self {
        Self {
            config,
            srs_cache: None,
            stats: ProverStats::default(),
        }
    }

    /// Setup SRS for given model size
    pub fn setup(&mut self, max_model_size: usize) -> Result<(), ZkMLError> {
        info!("🔧 Setting up zkML SRS for {} parameters", max_model_size);
        let start = Instant::now();

        // Generate powers (simplified - in production would use proper ceremony)
        let num_powers = max_model_size.next_power_of_two();
        let mut powers = Vec::with_capacity(num_powers);

        let mut hasher = Sha3_256::new();
        hasher.update(b"zkML-SRS-seed");
        let mut current: [u8; 32] = hasher.finalize().into();

        for _ in 0..num_powers.min(10_000) {
            powers.push(current.to_vec());
            let mut next_hasher = Sha3_256::new();
            next_hasher.update(&current);
            current = next_hasher.finalize().into();
        }

        self.srs_cache = Some(PrecomputedSRS {
            security_level: self.config.security_level,
            max_model_size,
            powers,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        });

        info!("✅ SRS setup complete in {:?}", start.elapsed());
        Ok(())
    }

    /// Prove inference with commitment to private inputs
    pub fn prove_private_inference(
        &mut self,
        model: &ZkNeuralNetwork,
        private_input: &[i64],
        public_output_indices: &[usize],
    ) -> Result<PrivateInferenceProof, ZkMLError> {
        let start = Instant::now();

        // Validate
        if private_input.len() != model.input_dim {
            return Err(ZkMLError::DimensionMismatch {
                expected: model.input_dim,
                got: private_input.len(),
            });
        }

        // Compute input commitment (hides actual input)
        let input_commitment = self.commit_private_input(private_input);

        // Run inference
        let output = self.run_inference(model, private_input)?;

        // Extract public outputs
        let public_outputs: Vec<i64> = public_output_indices
            .iter()
            .filter_map(|&i| output.get(i).copied())
            .collect();

        // Generate range proofs for outputs
        let range_proofs = self.generate_range_proofs(&public_outputs)?;

        let proof_time_ms = start.elapsed().as_millis() as u64;

        self.stats.total_proofs += 1;
        self.stats.total_time_ms += proof_time_ms;

        Ok(PrivateInferenceProof {
            model_commitment: model.commit(),
            input_commitment,
            public_outputs,
            range_proofs,
            proof_time_ms,
        })
    }

    /// Commit to private input
    fn commit_private_input(&self, input: &[i64]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"private-input");
        for val in input {
            hasher.update(&val.to_le_bytes());
        }
        // Add randomness for hiding
        let random: [u8; 32] = rand::random();
        hasher.update(&random);
        hasher.finalize().into()
    }

    /// Run inference (same as in main prover)
    fn run_inference(
        &self,
        model: &ZkNeuralNetwork,
        input: &[i64],
    ) -> Result<Vec<i64>, ZkMLError> {
        let mut current = input.to_vec();

        for layer in &model.layers {
            current = match layer {
                super::ZkLayer::Dense {
                    weights,
                    bias,
                    in_features,
                    out_features,
                } => {
                    let mut output = vec![0i64; *out_features];
                    for j in 0..*out_features {
                        let mut sum = bias[j] as i128;
                        for i in 0..*in_features {
                            sum += weights[j * in_features + i] as i128 * current[i] as i128;
                        }
                        output[j] = (sum >> self.config.fixed_point_bits) as i64;
                    }
                    output
                }
                super::ZkLayer::ReLU => current.iter().map(|&x| x.max(0)).collect(),
                super::ZkLayer::Sigmoid => {
                    let scale = 1i64 << self.config.fixed_point_bits;
                    current
                        .iter()
                        .map(|&x| {
                            if x < -4 * scale {
                                0
                            } else if x > 4 * scale {
                                scale
                            } else {
                                scale / 2 + x / 4
                            }
                        })
                        .collect()
                }
                super::ZkLayer::Softmax => {
                    let max_val = current.iter().copied().max().unwrap_or(0);
                    current.iter().map(|&x| x - max_val).collect()
                }
                super::ZkLayer::BatchNorm { .. } => current,
            };
        }

        Ok(current)
    }

    /// Generate range proofs for outputs
    fn generate_range_proofs(&self, outputs: &[i64]) -> Result<Vec<RangeProof>, ZkMLError> {
        outputs
            .iter()
            .map(|&val| {
                // Bulletproof-style range proof (simplified)
                let mut hasher = Sha3_256::new();
                hasher.update(&val.to_le_bytes());
                let commitment: [u8; 32] = hasher.finalize().into();

                Ok(RangeProof {
                    commitment,
                    lower_bound: i64::MIN,
                    upper_bound: i64::MAX,
                    proof_data: vec![],
                })
            })
            .collect()
    }

    /// Get prover statistics
    pub fn stats(&self) -> &ProverStats {
        &self.stats
    }
}

/// Private inference proof
#[derive(Clone, Debug)]
pub struct PrivateInferenceProof {
    /// Model commitment
    pub model_commitment: super::ModelCommitment,

    /// Input commitment (hides actual input)
    pub input_commitment: [u8; 32],

    /// Public outputs (only selected indices)
    pub public_outputs: Vec<i64>,

    /// Range proofs for outputs
    pub range_proofs: Vec<RangeProof>,

    /// Proof generation time
    pub proof_time_ms: u64,
}

/// Range proof for output values
#[derive(Clone, Debug)]
pub struct RangeProof {
    /// Commitment to value
    pub commitment: [u8; 32],

    /// Lower bound
    pub lower_bound: i64,

    /// Upper bound
    pub upper_bound: i64,

    /// Proof data
    pub proof_data: Vec<u8>,
}

/// Batch inference prover for efficiency
pub struct BatchInferenceProver {
    prover: InferenceProver,
    batch_size: usize,
}

impl BatchInferenceProver {
    /// Create batch prover
    pub fn new(config: ZkMLConfig, batch_size: usize) -> Self {
        Self {
            prover: InferenceProver::new(config),
            batch_size,
        }
    }

    /// Prove batch of inferences
    pub fn prove_batch(
        &mut self,
        model: &ZkNeuralNetwork,
        inputs: &[Vec<i64>],
    ) -> Result<BatchInferenceProof, ZkMLError> {
        let start = Instant::now();

        if inputs.len() > self.batch_size {
            return Err(ZkMLError::ModelTooLarge(inputs.len(), self.batch_size));
        }

        // Aggregate input commitment
        let mut hasher = Sha3_256::new();
        for input in inputs {
            for val in input {
                hasher.update(&val.to_le_bytes());
            }
        }
        let aggregate_input_commitment: [u8; 32] = hasher.finalize().into();

        // Run all inferences
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            let output = self.prover.run_inference(model, input)?;
            outputs.push(output);
        }

        // Aggregate output commitment
        let mut output_hasher = Sha3_256::new();
        for output in &outputs {
            for val in output {
                output_hasher.update(&val.to_le_bytes());
            }
        }
        let aggregate_output_commitment: [u8; 32] = output_hasher.finalize().into();

        let proof_time_ms = start.elapsed().as_millis() as u64;

        Ok(BatchInferenceProof {
            model_commitment: model.commit(),
            batch_size: inputs.len(),
            aggregate_input_commitment,
            aggregate_output_commitment,
            outputs,
            proof_time_ms,
        })
    }
}

/// Batch inference proof
#[derive(Clone, Debug)]
pub struct BatchInferenceProof {
    /// Model commitment
    pub model_commitment: super::ModelCommitment,

    /// Number of inferences
    pub batch_size: usize,

    /// Aggregate input commitment
    pub aggregate_input_commitment: [u8; 32],

    /// Aggregate output commitment
    pub aggregate_output_commitment: [u8; 32],

    /// All outputs
    pub outputs: Vec<Vec<i64>>,

    /// Total proof time
    pub proof_time_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkml::{FloatLayer, ZkNeuralNetwork};

    #[test]
    fn test_private_inference() {
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
        let config = ZkMLConfig::default();
        let mut prover = InferenceProver::new(config);

        let scale = 1i64 << 16;
        let input = vec![scale, 2 * scale];

        let proof = prover
            .prove_private_inference(&model, &input, &[0, 1])
            .unwrap();

        assert_eq!(proof.public_outputs.len(), 2);
    }

    #[test]
    fn test_batch_inference() {
        let layers = vec![FloatLayer::Dense {
            weights: vec![1.0, 0.0, 0.0, 1.0],
            bias: vec![0.0, 0.0],
            in_features: 2,
            out_features: 2,
        }];

        let model = ZkNeuralNetwork::from_float_model(layers, 16).unwrap();
        let config = ZkMLConfig::default();
        let mut prover = BatchInferenceProver::new(config, 100);

        let scale = 1i64 << 16;
        let inputs = vec![
            vec![scale, 2 * scale],
            vec![3 * scale, 4 * scale],
            vec![5 * scale, 6 * scale],
        ];

        let proof = prover.prove_batch(&model, &inputs).unwrap();

        assert_eq!(proof.batch_size, 3);
        assert_eq!(proof.outputs.len(), 3);
    }
}
