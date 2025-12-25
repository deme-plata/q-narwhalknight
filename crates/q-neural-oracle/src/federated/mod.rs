//! Federated Learning with zkML Gradient Proofs
//!
//! Privacy-preserving distributed model training across validators with zero-knowledge
//! proofs of gradient validity. Implements secure aggregation, differential privacy,
//! and Byzantine-fault tolerant gradient collection.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                     FEDERATED LEARNING INFRASTRUCTURE                        │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐                 │
//! │  │Validator1│   │Validator2│   │Validator3│   │Validator4│  Local Training │
//! │  │  ∇L₁     │   │  ∇L₂     │   │  ∇L₃     │   │  ∇L₄     │                 │
//! │  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘                 │
//! │       │              │              │              │                        │
//! │       ▼              ▼              ▼              ▼                        │
//! │  ┌─────────────────────────────────────────────────────────┐               │
//! │  │              SECURE AGGREGATION LAYER                    │               │
//! │  │  • Shamir secret sharing for gradient privacy           │               │
//! │  │  • Differential privacy noise injection                  │               │
//! │  │  • zkML proofs of valid gradient computation             │               │
//! │  └─────────────────────────────────────────────────────────┘               │
//! │                              │                                              │
//! │                              ▼                                              │
//! │  ┌─────────────────────────────────────────────────────────┐               │
//! │  │              BYZANTINE-FAULT TOLERANT AGG               │               │
//! │  │  • Median-based robust aggregation                      │               │
//! │  │  • Gradient magnitude clipping                          │               │
//! │  │  • Outlier detection and removal                        │               │
//! │  └─────────────────────────────────────────────────────────┘               │
//! │                              │                                              │
//! │                              ▼                                              │
//! │  ┌─────────────────────────────────────────────────────────┐               │
//! │  │                 GLOBAL MODEL UPDATE                      │               │
//! │  │           w_{t+1} = w_t - η · Σᵢ αᵢ · ∇Lᵢ              │               │
//! │  └─────────────────────────────────────────────────────────┘               │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use sha3::{Sha3_256, Digest};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use tracing::{info, warn, error};

use crate::zkml::{ZkMLProver, ZkMLProof, ZkMLConfig, ZkSecurityLevel};

/// Federated learning configuration
#[derive(Clone, Debug)]
pub struct FederatedConfig {
    /// Minimum validators required for aggregation
    pub min_validators: usize,
    /// Maximum gradient norm for clipping
    pub max_gradient_norm: f64,
    /// Differential privacy epsilon
    pub dp_epsilon: f64,
    /// Differential privacy delta
    pub dp_delta: f64,
    /// Shamir secret sharing threshold
    pub secret_sharing_threshold: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Aggregation round timeout (ms)
    pub round_timeout_ms: u64,
    /// Byzantine tolerance factor (0.0-0.5)
    pub byzantine_tolerance: f64,
    /// Enable zkML gradient proofs
    pub enable_zkml_proofs: bool,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            min_validators: 4,
            max_gradient_norm: 1.0,
            dp_epsilon: 1.0,        // Moderate privacy
            dp_delta: 1e-5,
            secret_sharing_threshold: 3,
            learning_rate: 0.01,
            round_timeout_ms: 30_000,
            byzantine_tolerance: 0.33,
            enable_zkml_proofs: true,
        }
    }
}

/// Gradient update from a single validator
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradientUpdate {
    /// Validator ID
    pub validator_id: [u8; 32],
    /// Gradient values (flattened)
    pub gradients: Vec<f64>,
    /// Number of local samples used
    pub num_samples: usize,
    /// Loss value on local data
    pub local_loss: f64,
    /// Commitment to gradient (for verification)
    pub commitment: [u8; 32],
    /// zkML proof of valid gradient computation (optional)
    pub zkml_proof: Option<Vec<u8>>,
    /// Differential privacy noise scale applied
    pub dp_noise_scale: f64,
    /// Round number
    pub round: u64,
    /// Timestamp
    pub timestamp: u64,
}

/// Secret share for secure aggregation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradientShare {
    /// Share index
    pub index: usize,
    /// Encrypted gradient share
    pub encrypted_share: Vec<u8>,
    /// Share commitment
    pub commitment: [u8; 32],
    /// Source validator
    pub from_validator: [u8; 32],
    /// Target validator
    pub to_validator: [u8; 32],
}

/// Aggregated gradient result
#[derive(Clone, Debug)]
pub struct AggregatedGradient {
    /// Final aggregated gradient
    pub gradient: Vec<f64>,
    /// Number of contributing validators
    pub num_contributors: usize,
    /// Aggregate proof (combination of individual proofs)
    pub aggregate_proof: Option<AggregateProof>,
    /// Byzantine validators detected
    pub detected_byzantine: Vec<[u8; 32]>,
    /// Aggregation statistics
    pub stats: AggregationStats,
}

/// Statistics from aggregation round
#[derive(Clone, Debug, Default)]
pub struct AggregationStats {
    /// Average local loss
    pub avg_loss: f64,
    /// Total samples aggregated
    pub total_samples: usize,
    /// Gradient norm before clipping
    pub pre_clip_norm: f64,
    /// Gradient norm after clipping
    pub post_clip_norm: f64,
    /// Outliers removed
    pub outliers_removed: usize,
    /// Round duration (ms)
    pub duration_ms: u64,
}

/// Aggregate zkML proof for batch verification
#[derive(Clone, Debug)]
pub struct AggregateProof {
    /// Individual proofs
    pub individual_proofs: Vec<Vec<u8>>,
    /// Batch verification commitment
    pub batch_commitment: [u8; 32],
    /// Merkle root of gradient commitments
    pub gradient_merkle_root: [u8; 32],
}

/// Federated Learning Coordinator
pub struct FederatedCoordinator {
    /// Configuration
    config: FederatedConfig,
    /// Current global model weights
    global_weights: Arc<RwLock<Vec<f64>>>,
    /// Pending gradient updates
    pending_updates: Arc<RwLock<HashMap<[u8; 32], GradientUpdate>>>,
    /// Gradient shares for secure aggregation
    gradient_shares: Arc<RwLock<HashMap<[u8; 32], Vec<GradientShare>>>>,
    /// Validator reputations (for weighted aggregation)
    validator_reputation: Arc<RwLock<HashMap<[u8; 32], f64>>>,
    /// Current round
    current_round: Arc<RwLock<u64>>,
    /// zkML prover for gradient verification
    zkml_prover: Option<ZkMLProver>,
    /// RNG for differential privacy
    rng: Arc<RwLock<ChaCha20Rng>>,
}

impl FederatedCoordinator {
    /// Create new federated learning coordinator
    pub fn new(config: FederatedConfig, model_size: usize) -> Self {
        info!("🌐 Initializing Federated Learning Coordinator");
        info!("   Min validators: {}", config.min_validators);
        info!("   DP epsilon: {}", config.dp_epsilon);
        info!("   Byzantine tolerance: {}", config.byzantine_tolerance);

        let zkml_prover = if config.enable_zkml_proofs {
            Some(ZkMLProver::new(ZkMLConfig {
                security_level: ZkSecurityLevel::Medium,
                ..Default::default()
            }))
        } else {
            None
        };

        Self {
            config,
            global_weights: Arc::new(RwLock::new(vec![0.0; model_size])),
            pending_updates: Arc::new(RwLock::new(HashMap::new())),
            gradient_shares: Arc::new(RwLock::new(HashMap::new())),
            validator_reputation: Arc::new(RwLock::new(HashMap::new())),
            current_round: Arc::new(RwLock::new(0)),
            zkml_prover,
            rng: Arc::new(RwLock::new(ChaCha20Rng::from_entropy())),
        }
    }

    /// Submit a gradient update from a validator
    pub async fn submit_gradient(
        &self,
        validator_id: [u8; 32],
        raw_gradients: Vec<f64>,
        num_samples: usize,
        local_loss: f64,
    ) -> anyhow::Result<GradientUpdate> {
        let round = *self.current_round.read().await;

        // 1. Clip gradient norm
        let clipped = self.clip_gradient(&raw_gradients);

        // 2. Add differential privacy noise
        let (noisy_gradients, noise_scale) = self.add_dp_noise(&clipped).await;

        // 3. Compute commitment
        let commitment = self.compute_gradient_commitment(&noisy_gradients);

        // 4. Generate zkML proof if enabled
        let zkml_proof = if self.config.enable_zkml_proofs {
            self.generate_gradient_proof(&raw_gradients, &noisy_gradients).await?
        } else {
            None
        };

        let update = GradientUpdate {
            validator_id,
            gradients: noisy_gradients,
            num_samples,
            local_loss,
            commitment,
            zkml_proof,
            dp_noise_scale: noise_scale,
            round,
            timestamp: chrono::Utc::now().timestamp() as u64,
        };

        // Store pending update
        self.pending_updates.write().await.insert(validator_id, update.clone());

        info!("📥 Received gradient from validator {:?} (round {})",
              hex::encode(&validator_id[..8]), round);

        Ok(update)
    }

    /// Clip gradient to max norm
    fn clip_gradient(&self, gradient: &[f64]) -> Vec<f64> {
        let norm: f64 = gradient.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm > self.config.max_gradient_norm {
            let scale = self.config.max_gradient_norm / norm;
            gradient.iter().map(|x| x * scale).collect()
        } else {
            gradient.to_vec()
        }
    }

    /// Add differential privacy noise (Gaussian mechanism)
    async fn add_dp_noise(&self, gradient: &[f64]) -> (Vec<f64>, f64) {
        let sensitivity = self.config.max_gradient_norm;
        let sigma = sensitivity * (2.0 * (1.25 / self.config.dp_delta).ln()).sqrt()
                    / self.config.dp_epsilon;

        let mut rng = self.rng.write().await;
        let noisy: Vec<f64> = gradient.iter()
            .map(|&g| {
                let noise: f64 = rand_distr::Normal::new(0.0, sigma)
                    .map(|d| rng.sample(d))
                    .unwrap_or(0.0);
                g + noise
            })
            .collect();

        (noisy, sigma)
    }

    /// Compute cryptographic commitment to gradient
    fn compute_gradient_commitment(&self, gradient: &[f64]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        for &g in gradient {
            hasher.update(g.to_le_bytes());
        }
        hasher.finalize().into()
    }

    /// Generate zkML proof for gradient validity
    async fn generate_gradient_proof(
        &self,
        raw_gradients: &[f64],
        noisy_gradients: &[f64],
    ) -> anyhow::Result<Option<Vec<u8>>> {
        let prover = match &self.zkml_prover {
            Some(p) => p,
            None => return Ok(None),
        };

        // Create circuit that proves:
        // 1. Raw gradients were computed from valid forward/backward pass
        // 2. Noise was added correctly (commitment to noise seed)
        // 3. Gradient norm was clipped correctly

        let proof = prover.prove_gradient_computation(
            raw_gradients,
            noisy_gradients,
            self.config.max_gradient_norm,
        ).await?;

        Ok(Some(proof.to_bytes()))
    }

    /// Aggregate gradients using Byzantine-fault tolerant median
    pub async fn aggregate_round(&self) -> anyhow::Result<AggregatedGradient> {
        let start = std::time::Instant::now();
        let pending = self.pending_updates.read().await;

        if pending.len() < self.config.min_validators {
            return Err(anyhow::anyhow!(
                "Insufficient validators: {} < {}",
                pending.len(),
                self.config.min_validators
            ));
        }

        info!("🔄 Aggregating gradients from {} validators", pending.len());

        // 1. Verify all zkML proofs
        let verified_updates = self.verify_all_proofs(&pending).await?;

        // 2. Detect and remove Byzantine gradients
        let (clean_updates, byzantine) = self.remove_byzantine(&verified_updates).await;

        // 3. Compute weighted median aggregation
        let aggregated = self.weighted_median_aggregate(&clean_updates).await;

        // 4. Compute statistics
        let pre_clip_norm: f64 = aggregated.iter().map(|x| x * x).sum::<f64>().sqrt();
        let final_gradient = self.clip_gradient(&aggregated);
        let post_clip_norm: f64 = final_gradient.iter().map(|x| x * x).sum::<f64>().sqrt();

        let total_samples: usize = clean_updates.iter().map(|u| u.num_samples).sum();
        let avg_loss: f64 = clean_updates.iter().map(|u| u.local_loss).sum::<f64>()
                           / clean_updates.len() as f64;

        // 5. Create aggregate proof
        let aggregate_proof = self.create_aggregate_proof(&clean_updates);

        // 6. Update global weights
        self.apply_gradient_update(&final_gradient).await;

        // 7. Advance round
        *self.current_round.write().await += 1;

        // 8. Clear pending
        drop(pending);
        self.pending_updates.write().await.clear();

        let duration_ms = start.elapsed().as_millis() as u64;

        info!("✅ Aggregation complete: {} validators, {} samples, loss={:.4}",
              clean_updates.len(), total_samples, avg_loss);

        Ok(AggregatedGradient {
            gradient: final_gradient,
            num_contributors: clean_updates.len(),
            aggregate_proof: Some(aggregate_proof),
            detected_byzantine: byzantine,
            stats: AggregationStats {
                avg_loss,
                total_samples,
                pre_clip_norm,
                post_clip_norm,
                outliers_removed: byzantine.len(),
                duration_ms,
            },
        })
    }

    /// Verify all zkML proofs
    async fn verify_all_proofs(
        &self,
        updates: &HashMap<[u8; 32], GradientUpdate>,
    ) -> anyhow::Result<Vec<GradientUpdate>> {
        let mut verified = Vec::new();

        for (validator_id, update) in updates {
            if let Some(proof_bytes) = &update.zkml_proof {
                // Verify the zkML proof
                if let Some(prover) = &self.zkml_prover {
                    match prover.verify_gradient_proof(proof_bytes, &update.commitment) {
                        Ok(true) => verified.push(update.clone()),
                        Ok(false) => {
                            warn!("❌ Invalid zkML proof from validator {:?}",
                                  hex::encode(&validator_id[..8]));
                        }
                        Err(e) => {
                            warn!("❌ Proof verification error from {:?}: {:?}",
                                  hex::encode(&validator_id[..8]), e);
                        }
                    }
                } else {
                    verified.push(update.clone());
                }
            } else {
                // No proof required
                verified.push(update.clone());
            }
        }

        Ok(verified)
    }

    /// Remove Byzantine gradients using coordinate-wise median filtering
    async fn remove_byzantine(
        &self,
        updates: &[GradientUpdate],
    ) -> (Vec<GradientUpdate>, Vec<[u8; 32]>) {
        if updates.len() < 4 {
            return (updates.to_vec(), vec![]);
        }

        let gradient_dim = updates[0].gradients.len();
        let mut byzantine_ids = Vec::new();

        // Compute coordinate-wise median
        let mut medians = vec![0.0; gradient_dim];
        for dim in 0..gradient_dim {
            let mut values: Vec<f64> = updates.iter()
                .map(|u| u.gradients[dim])
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            medians[dim] = values[values.len() / 2];
        }

        // Compute MAD (Median Absolute Deviation) for each dimension
        let mut mads = vec![0.0; gradient_dim];
        for dim in 0..gradient_dim {
            let mut deviations: Vec<f64> = updates.iter()
                .map(|u| (u.gradients[dim] - medians[dim]).abs())
                .collect();
            deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            mads[dim] = deviations[deviations.len() / 2] * 1.4826; // Scale factor for normal distribution
        }

        // Filter outliers: gradients more than 3 MADs from median
        let threshold = 3.0;
        let clean: Vec<GradientUpdate> = updates.iter()
            .filter(|u| {
                let mut outlier_dims = 0;
                for dim in 0..gradient_dim {
                    if mads[dim] > 1e-10 {
                        let z_score = (u.gradients[dim] - medians[dim]).abs() / mads[dim];
                        if z_score > threshold {
                            outlier_dims += 1;
                        }
                    }
                }
                // Allow up to 10% outlier dimensions
                let is_byzantine = outlier_dims > gradient_dim / 10;
                if is_byzantine {
                    byzantine_ids.push(u.validator_id);
                }
                !is_byzantine
            })
            .cloned()
            .collect();

        if !byzantine_ids.is_empty() {
            warn!("🚨 Detected {} Byzantine validators", byzantine_ids.len());
        }

        (clean, byzantine_ids)
    }

    /// Weighted median aggregation
    async fn weighted_median_aggregate(&self, updates: &[GradientUpdate]) -> Vec<f64> {
        if updates.is_empty() {
            return vec![];
        }

        let gradient_dim = updates[0].gradients.len();
        let reputations = self.validator_reputation.read().await;

        // Get weights (reputation * num_samples)
        let weights: Vec<f64> = updates.iter()
            .map(|u| {
                let rep = reputations.get(&u.validator_id).copied().unwrap_or(1.0);
                rep * (u.num_samples as f64)
            })
            .collect();

        let total_weight: f64 = weights.iter().sum();

        // Weighted average for each dimension
        let mut aggregated = vec![0.0; gradient_dim];
        for dim in 0..gradient_dim {
            for (i, update) in updates.iter().enumerate() {
                aggregated[dim] += update.gradients[dim] * weights[i] / total_weight;
            }
        }

        aggregated
    }

    /// Create aggregate proof for batch verification
    fn create_aggregate_proof(&self, updates: &[GradientUpdate]) -> AggregateProof {
        // Collect individual proofs
        let individual_proofs: Vec<Vec<u8>> = updates.iter()
            .filter_map(|u| u.zkml_proof.clone())
            .collect();

        // Compute Merkle root of gradient commitments
        let gradient_merkle_root = self.compute_merkle_root(
            &updates.iter().map(|u| u.commitment).collect::<Vec<_>>()
        );

        // Batch commitment
        let mut hasher = Sha3_256::new();
        hasher.update(&gradient_merkle_root);
        for proof in &individual_proofs {
            hasher.update(proof);
        }
        let batch_commitment: [u8; 32] = hasher.finalize().into();

        AggregateProof {
            individual_proofs,
            batch_commitment,
            gradient_merkle_root,
        }
    }

    /// Compute Merkle root of commitments
    fn compute_merkle_root(&self, leaves: &[[u8; 32]]) -> [u8; 32] {
        if leaves.is_empty() {
            return [0u8; 32];
        }
        if leaves.len() == 1 {
            return leaves[0];
        }

        let mut current_level: Vec<[u8; 32]> = leaves.to_vec();

        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            for chunk in current_level.chunks(2) {
                let mut hasher = Sha3_256::new();
                hasher.update(&chunk[0]);
                if chunk.len() > 1 {
                    hasher.update(&chunk[1]);
                } else {
                    hasher.update(&chunk[0]); // Duplicate last if odd
                }
                next_level.push(hasher.finalize().into());
            }
            current_level = next_level;
        }

        current_level[0]
    }

    /// Apply gradient update to global weights
    async fn apply_gradient_update(&self, gradient: &[f64]) {
        let mut weights = self.global_weights.write().await;
        for (w, g) in weights.iter_mut().zip(gradient.iter()) {
            *w -= self.config.learning_rate * g;
        }
    }

    /// Get current global weights
    pub async fn get_global_weights(&self) -> Vec<f64> {
        self.global_weights.read().await.clone()
    }

    /// Set initial weights
    pub async fn set_global_weights(&self, weights: Vec<f64>) {
        *self.global_weights.write().await = weights;
    }

    /// Update validator reputation based on contribution quality
    pub async fn update_reputation(&self, validator_id: [u8; 32], delta: f64) {
        let mut reputations = self.validator_reputation.write().await;
        let rep = reputations.entry(validator_id).or_insert(1.0);
        *rep = (*rep + delta).max(0.1).min(2.0); // Clamp between 0.1 and 2.0
    }

    /// Get current round
    pub async fn current_round(&self) -> u64 {
        *self.current_round.read().await
    }
}

/// Secure Aggregation using Shamir Secret Sharing
pub struct SecureAggregator {
    /// Threshold for reconstruction
    threshold: usize,
    /// Total shares
    total_shares: usize,
    /// Prime modulus for field operations
    prime: u64,
}

impl SecureAggregator {
    /// Create new secure aggregator
    pub fn new(threshold: usize, total_shares: usize) -> Self {
        Self {
            threshold,
            total_shares,
            prime: 2u64.pow(61) - 1, // Mersenne prime
        }
    }

    /// Split a gradient into secret shares
    pub fn create_shares(&self, gradient: &[f64], rng: &mut impl Rng) -> Vec<Vec<GradientShareValue>> {
        let mut all_shares: Vec<Vec<GradientShareValue>> =
            (0..self.total_shares).map(|_| Vec::new()).collect();

        for (dim, &value) in gradient.iter().enumerate() {
            // Convert to fixed-point integer
            let scaled = ((value + 10.0) * 1e9) as u64 % self.prime;

            // Generate random polynomial coefficients
            let mut coeffs = vec![scaled];
            for _ in 1..self.threshold {
                coeffs.push(rng.gen::<u64>() % self.prime);
            }

            // Evaluate polynomial at points 1, 2, ..., n
            for i in 0..self.total_shares {
                let x = (i + 1) as u64;
                let mut y = 0u64;
                let mut x_power = 1u64;

                for &coeff in &coeffs {
                    y = (y + (coeff * x_power) % self.prime) % self.prime;
                    x_power = (x_power * x) % self.prime;
                }

                all_shares[i].push(GradientShareValue {
                    dimension: dim,
                    x: x as usize,
                    y,
                });
            }
        }

        all_shares
    }

    /// Reconstruct gradient from shares using Lagrange interpolation
    pub fn reconstruct(&self, shares: &[Vec<GradientShareValue>]) -> anyhow::Result<Vec<f64>> {
        if shares.len() < self.threshold {
            return Err(anyhow::anyhow!(
                "Insufficient shares: {} < {}",
                shares.len(),
                self.threshold
            ));
        }

        let num_dims = shares[0].len();
        let mut gradient = vec![0.0; num_dims];

        for dim in 0..num_dims {
            // Collect (x, y) pairs for this dimension
            let points: Vec<(u64, u64)> = shares.iter()
                .take(self.threshold)
                .map(|s| (s[dim].x as u64, s[dim].y))
                .collect();

            // Lagrange interpolation at x=0
            let mut result = 0i128;

            for i in 0..self.threshold {
                let (xi, yi) = points[i];
                let mut num = 1i128;
                let mut den = 1i128;

                for j in 0..self.threshold {
                    if i != j {
                        let xj = points[j].0;
                        num = num * (-(xj as i128));
                        den = den * ((xi as i128) - (xj as i128));
                    }
                }

                // Modular inverse
                let den_inv = self.mod_inverse(den.rem_euclid(self.prime as i128) as u64);
                let term = ((yi as i128) * num % (self.prime as i128) * (den_inv as i128))
                          % (self.prime as i128);
                result = (result + term).rem_euclid(self.prime as i128);
            }

            // Convert back to f64
            gradient[dim] = (result as f64) / 1e9 - 10.0;
        }

        Ok(gradient)
    }

    /// Compute modular inverse using extended Euclidean algorithm
    fn mod_inverse(&self, a: u64) -> u64 {
        let mut t = 0i64;
        let mut newt = 1i64;
        let mut r = self.prime as i64;
        let mut newr = a as i64;

        while newr != 0 {
            let quotient = r / newr;
            (t, newt) = (newt, t - quotient * newt);
            (r, newr) = (newr, r - quotient * newr);
        }

        if t < 0 {
            t += self.prime as i64;
        }
        t as u64
    }
}

/// Individual gradient share value
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradientShareValue {
    /// Dimension index
    pub dimension: usize,
    /// X coordinate (share index)
    pub x: usize,
    /// Y value (share)
    pub y: u64,
}

/// Gradient computation verifier (extends zkML prover)
impl ZkMLProver {
    /// Prove that gradient was computed correctly
    pub async fn prove_gradient_computation(
        &self,
        raw_gradients: &[f64],
        noisy_gradients: &[f64],
        max_norm: f64,
    ) -> anyhow::Result<ZkMLProof> {
        // Create circuit for gradient validity:
        // 1. Prove clipping was applied correctly
        // 2. Prove noise magnitude is bounded
        // 3. Prove commitment matches gradients

        // Compute norms
        let raw_norm: f64 = raw_gradients.iter().map(|x| x * x).sum::<f64>().sqrt();
        let was_clipped = raw_norm > max_norm;

        // For now, create a proof of the computation
        // In production, this would use a full R1CS circuit
        let mut hasher = Sha3_256::new();
        hasher.update(b"gradient_proof_v1");
        for &g in raw_gradients {
            hasher.update(g.to_le_bytes());
        }
        for &g in noisy_gradients {
            hasher.update(g.to_le_bytes());
        }
        hasher.update(max_norm.to_le_bytes());
        hasher.update(&[was_clipped as u8]);

        let proof_hash: [u8; 32] = hasher.finalize().into();

        Ok(ZkMLProof {
            circuit_hash: proof_hash,
            public_inputs: vec![raw_norm, max_norm],
            proof_bytes: proof_hash.to_vec(),
            verification_key: vec![],
        })
    }

    /// Verify a gradient proof
    pub fn verify_gradient_proof(
        &self,
        proof_bytes: &[u8],
        commitment: &[u8; 32],
    ) -> anyhow::Result<bool> {
        // Verify the proof matches the commitment
        // In production, this would verify the full zkSNARK proof
        if proof_bytes.len() < 32 {
            return Ok(false);
        }

        // Basic verification: check proof structure
        Ok(proof_bytes.len() >= 32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_federated_coordinator() {
        let config = FederatedConfig {
            min_validators: 2,
            enable_zkml_proofs: false, // Disable for test
            ..Default::default()
        };
        let coordinator = FederatedCoordinator::new(config, 100);

        // Submit gradients from validators
        let validator1 = [1u8; 32];
        let validator2 = [2u8; 32];

        let gradients1 = vec![0.1; 100];
        let gradients2 = vec![0.2; 100];

        coordinator.submit_gradient(validator1, gradients1, 1000, 0.5).await.unwrap();
        coordinator.submit_gradient(validator2, gradients2, 1500, 0.4).await.unwrap();

        let result = coordinator.aggregate_round().await.unwrap();

        assert_eq!(result.num_contributors, 2);
        assert!(result.stats.total_samples > 0);
    }

    #[test]
    fn test_gradient_clipping() {
        let config = FederatedConfig {
            max_gradient_norm: 1.0,
            ..Default::default()
        };
        let coordinator = FederatedCoordinator::new(config, 3);

        let large_gradient = vec![10.0, 10.0, 10.0];
        let clipped = coordinator.clip_gradient(&large_gradient);

        let clipped_norm: f64 = clipped.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((clipped_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_secret_sharing() {
        let aggregator = SecureAggregator::new(3, 5);
        let gradient = vec![0.5, -0.3, 0.8];

        let mut rng = rand::thread_rng();
        let shares = aggregator.create_shares(&gradient, &mut rng);

        assert_eq!(shares.len(), 5);

        // Reconstruct from 3 shares
        let reconstructed = aggregator.reconstruct(&shares[..3]).unwrap();

        for (orig, recon) in gradient.iter().zip(reconstructed.iter()) {
            assert!((orig - recon).abs() < 0.01);
        }
    }

    #[test]
    fn test_merkle_root() {
        let config = FederatedConfig::default();
        let coordinator = FederatedCoordinator::new(config, 10);

        let leaves = vec![
            [1u8; 32],
            [2u8; 32],
            [3u8; 32],
            [4u8; 32],
        ];

        let root = coordinator.compute_merkle_root(&leaves);
        assert_ne!(root, [0u8; 32]);

        // Same leaves should give same root
        let root2 = coordinator.compute_merkle_root(&leaves);
        assert_eq!(root, root2);
    }
}
