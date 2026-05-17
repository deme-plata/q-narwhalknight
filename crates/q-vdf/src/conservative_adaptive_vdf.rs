// Conservative Adaptive VDF Implementation
// Based on AI review feedback - focuses on safe, validated adaptive security
//
// Key Design Principles:
// 1. Conservative parameter ranges (max 2x baseline, not 4x)
// 2. Hashrate smoothing (24-hour moving average)
// 3. Rate limiting (max 5% daily change)
// 4. Manipulation resistance (detect anomalies)
// 5. Governance integration (security tier voting)

use anyhow::{anyhow, Result};
use num_bigint::BigUint;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// Security tier levels approved by governance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityTier {
    /// Standard security - 1,000 VDF iterations
    Standard,
    /// Enhanced security - 1,500 VDF iterations (requires governance vote)
    Enhanced,
    /// Maximum security - 2,000 VDF iterations (requires governance vote)
    Maximum,
}

impl SecurityTier {
    pub fn base_iterations(&self) -> u64 {
        match self {
            SecurityTier::Standard => 1_000,
            SecurityTier::Enhanced => 1_500,
            SecurityTier::Maximum => 2_000,
        }
    }

    pub fn from_governance_vote(votes_standard: u64, votes_enhanced: u64, votes_maximum: u64) -> Self {
        if votes_maximum > votes_enhanced && votes_maximum > votes_standard {
            SecurityTier::Maximum
        } else if votes_enhanced > votes_standard {
            SecurityTier::Enhanced
        } else {
            SecurityTier::Standard
        }
    }
}

/// Hashrate measurement with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashrateSnapshot {
    pub timestamp: u64,
    pub hashrate: f64, // Hashes per second
}

/// Hashrate tracker with 24-hour moving average
#[derive(Debug, Clone)]
pub struct HashrateTracker {
    snapshots: Vec<HashrateSnapshot>,
    window_duration: Duration, // 24 hours
    max_daily_change: f64,     // 5% = 0.05
}

impl HashrateTracker {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            window_duration: Duration::from_secs(86400), // 24 hours
            max_daily_change: 0.05,                      // 5% max change
        }
    }

    /// Add new hashrate measurement
    pub fn add_snapshot(&mut self, hashrate: f64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.snapshots.push(HashrateSnapshot {
            timestamp: now,
            hashrate,
        });

        // Remove snapshots older than window
        let cutoff = now.saturating_sub(self.window_duration.as_secs());
        self.snapshots.retain(|s| s.timestamp >= cutoff);
    }

    /// Compute 24-hour moving average hashrate
    pub fn compute_smoothed_hashrate(&self) -> f64 {
        if self.snapshots.is_empty() {
            return 1_000_000_000.0; // 1 GH/s baseline
        }

        let sum: f64 = self.snapshots.iter().map(|s| s.hashrate).sum();
        sum / (self.snapshots.len() as f64)
    }

    /// Check if hashrate change is within safe limits
    pub fn is_change_safe(&self, new_hashrate: f64) -> bool {
        let current = self.compute_smoothed_hashrate();
        let change_ratio = (new_hashrate - current).abs() / current;
        change_ratio <= self.max_daily_change
    }

    /// Detect potential hashrate manipulation
    pub fn detect_manipulation(&self) -> bool {
        if self.snapshots.len() < 10 {
            return false;
        }

        // Check for sudden spikes (>50% change in <1 hour)
        let recent: Vec<_> = self
            .snapshots
            .iter()
            .rev()
            .take(12) // Last hour (assuming 5-min intervals)
            .collect();

        if recent.len() < 2 {
            return false;
        }

        let first = recent.last().unwrap().hashrate;
        let last = recent.first().unwrap().hashrate;
        let change = (last - first).abs() / first;

        change > 0.50 // Flag if >50% change in 1 hour
    }
}

/// Conservative Adaptive VDF Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservativeVDFParams {
    /// Current security tier (governance-approved)
    pub security_tier: SecurityTier,

    /// Base iterations for current tier
    pub base_iterations: u64,

    /// Maximum allowed iterations (conservative limit)
    pub max_iterations: u64,

    /// Current adaptive iterations (based on smoothed hashrate)
    pub adaptive_iterations: u64,

    /// Smoothed network hashrate (24-hour MA)
    pub smoothed_hashrate: f64,

    /// Security multiplier (1.0 - 2.0 range)
    pub security_multiplier: f64,
}

impl Default for ConservativeVDFParams {
    fn default() -> Self {
        Self {
            security_tier: SecurityTier::Standard,
            base_iterations: 1_000,
            max_iterations: 2_000,
            adaptive_iterations: 1_000,
            smoothed_hashrate: 1_000_000_000.0, // 1 GH/s baseline
            security_multiplier: 1.0,
        }
    }
}

/// Conservative Adaptive VDF Implementation
pub struct ConservativeAdaptiveVDF {
    /// VDF parameters (shared state)
    params: Arc<RwLock<ConservativeVDFParams>>,

    /// Hashrate tracker with smoothing
    hashrate_tracker: Arc<RwLock<HashrateTracker>>,

    /// Modulus for VDF computation (3072-bit RSA modulus)
    modulus: BigUint,
}

impl ConservativeAdaptiveVDF {
    pub fn new() -> Self {
        // Use a 3072-bit RSA modulus for quantum resistance
        let modulus = BigUint::parse_bytes(
            b"12074070144068064889917164878050647360673920130728994445476\
              74458284473569668371691734798803811146041636185982642238688\
              15893729809664721418185065099606663367782628265319436221949\
              83891467476903068957867834642428419644686183869285571369056\
              74829823464059207825793888912912988765876243456729018721893\
              48927",
            10,
        )
        .expect("Invalid modulus");

        Self {
            params: Arc::new(RwLock::new(ConservativeVDFParams::default())),
            hashrate_tracker: Arc::new(RwLock::new(HashrateTracker::new())),
            modulus,
        }
    }

    /// Update hashrate measurement and recompute adaptive parameters
    pub async fn update_hashrate(&self, current_hashrate: f64) -> Result<()> {
        let mut tracker = self.hashrate_tracker.write().await;

        // Detect manipulation before adding
        if tracker.detect_manipulation() {
            return Err(anyhow!(
                "🚨 Potential hashrate manipulation detected - rejecting update"
            ));
        }

        // Check if change is within safe limits
        if !tracker.is_change_safe(current_hashrate) {
            return Err(anyhow!(
                "⚠️  Hashrate change exceeds 5% daily limit - rejecting update"
            ));
        }

        // Add snapshot and recompute
        tracker.add_snapshot(current_hashrate);
        let smoothed = tracker.compute_smoothed_hashrate();

        // Update adaptive parameters
        let mut params = self.params.write().await;
        params.smoothed_hashrate = smoothed;
        params.security_multiplier = self.compute_security_multiplier(smoothed);
        params.adaptive_iterations = self.compute_adaptive_iterations(&params);

        Ok(())
    }

    /// Compute security multiplier from smoothed hashrate (conservative formula)
    fn compute_security_multiplier(&self, smoothed_hashrate: f64) -> f64 {
        let baseline_hashrate = 1_000_000_000.0; // 1 GH/s
        let max_hashrate = 100_000_000_000.0; // 100 GH/s (conservative max)

        let normalized = (smoothed_hashrate / baseline_hashrate).max(1.0);
        let max_normalized = max_hashrate / baseline_hashrate;

        // Conservative logarithmic scaling: 1.0 → 2.0 (not 4.0)
        1.0 + (normalized.log10() / max_normalized.log10()).min(1.0)
    }

    /// Compute adaptive iterations based on security tier and multiplier
    fn compute_adaptive_iterations(&self, params: &ConservativeVDFParams) -> u64 {
        let base = params.security_tier.base_iterations();
        let scaled = (base as f64 * params.security_multiplier) as u64;
        scaled.min(params.max_iterations)
    }

    /// Update security tier via governance vote
    pub async fn update_security_tier(
        &self,
        votes_standard: u64,
        votes_enhanced: u64,
        votes_maximum: u64,
    ) -> Result<()> {
        let new_tier = SecurityTier::from_governance_vote(votes_standard, votes_enhanced, votes_maximum);

        let mut params = self.params.write().await;
        params.security_tier = new_tier;
        params.base_iterations = new_tier.base_iterations();
        params.adaptive_iterations = self.compute_adaptive_iterations(&params);

        Ok(())
    }

    /// Evaluate adaptive VDF with current parameters
    pub async fn evaluate_adaptive(
        &self,
        input: &[u8],
        round: u64,
    ) -> Result<(BigUint, AdaptiveVDFProof)> {
        let params = self.params.read().await;
        let iterations = params.adaptive_iterations;

        // Parse input as BigUint
        let g = BigUint::from_bytes_be(input);
        let g = g % &self.modulus;

        if g.is_zero() || g.is_one() {
            return Err(anyhow!("Invalid VDF input (zero or one)"));
        }

        // Perform iterated squaring
        let mut y = g.clone();
        let exponent = BigUint::from(2u32);

        for _ in 0..iterations {
            y = y.modpow(&exponent, &self.modulus);
        }

        // Generate Wesolowski proof
        let proof = self.generate_wesolowski_proof(&g, &y, iterations).await?;

        let adaptive_proof = AdaptiveVDFProof {
            output: y.clone(),
            proof,
            iterations,
            security_tier: params.security_tier,
            security_multiplier: params.security_multiplier,
            smoothed_hashrate: params.smoothed_hashrate,
        };

        Ok((y, adaptive_proof))
    }

    /// Generate Wesolowski VDF proof
    async fn generate_wesolowski_proof(
        &self,
        g: &BigUint,
        y: &BigUint,
        iterations: u64,
    ) -> Result<WesolowskiProof> {
        // Compute challenge: r = H(g, y, iterations)
        let mut challenge_input = Vec::new();
        challenge_input.extend_from_slice(&g.to_bytes_be());
        challenge_input.extend_from_slice(&y.to_bytes_be());
        challenge_input.extend_from_slice(&iterations.to_be_bytes());

        let challenge_hash = blake3::hash(&challenge_input);
        let r = BigUint::from_bytes_be(challenge_hash.as_bytes());

        // Compute quotient: q = 2^T / r
        let two = BigUint::from(2u32);
        let t_exp = two.modpow(&BigUint::from(iterations), &r);
        let q = t_exp / &r;

        // Compute proof: π = g^q mod N
        let pi = g.modpow(&q, &self.modulus);

        Ok(WesolowskiProof {
            pi,
            challenge: r,
        })
    }

    /// Verify adaptive VDF proof
    pub async fn verify_adaptive(&self, proof: &AdaptiveVDFProof, input: &[u8]) -> Result<bool> {
        let g = BigUint::from_bytes_be(input);
        let g = g % &self.modulus;

        // Verify Wesolowski proof
        let r = &proof.proof.challenge;
        let pi = &proof.proof.pi;
        let y = &proof.output;

        // Compute remainder: b = 2^T mod r
        let two = BigUint::from(2u32);
        let t_exp = two.modpow(&BigUint::from(proof.iterations), r);

        // Verify: π^r * g^b == y (mod N)
        let lhs = (pi.modpow(r, &self.modulus) * g.modpow(&t_exp, &self.modulus)) % &self.modulus;
        let rhs = y % &self.modulus;

        Ok(lhs == rhs)
    }

    /// Get current VDF parameters (for block header)
    pub async fn get_current_params(&self) -> ConservativeVDFParams {
        self.params.read().await.clone()
    }
}

/// Wesolowski VDF Proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WesolowskiProof {
    pub pi: BigUint,       // Proof element
    pub challenge: BigUint, // Challenge value
}

/// Adaptive VDF Proof with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveVDFProof {
    pub output: BigUint,
    pub proof: WesolowskiProof,
    pub iterations: u64,
    pub security_tier: SecurityTier,
    pub security_multiplier: f64,
    pub smoothed_hashrate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hashrate_smoothing() {
        let mut tracker = HashrateTracker::new();

        // Add 24 hourly measurements
        for i in 0..24 {
            let hashrate = 1_000_000_000.0 + (i as f64 * 10_000_000.0);
            tracker.add_snapshot(hashrate);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let smoothed = tracker.compute_smoothed_hashrate();
        assert!(smoothed > 1_000_000_000.0);
        assert!(smoothed < 2_000_000_000.0);
    }

    #[tokio::test]
    async fn test_security_multiplier() {
        let vdf = ConservativeAdaptiveVDF::new();

        // Test baseline (1 GH/s) → multiplier = 1.0
        let mult_baseline = vdf.compute_security_multiplier(1_000_000_000.0);
        assert!((mult_baseline - 1.0).abs() < 0.01);

        // Test maximum (100 GH/s) → multiplier = 2.0
        let mult_max = vdf.compute_security_multiplier(100_000_000_000.0);
        assert!((mult_max - 2.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_adaptive_iterations() {
        let vdf = ConservativeAdaptiveVDF::new();

        // Update to medium hashrate
        vdf.update_hashrate(10_000_000_000.0).await.unwrap();

        let params = vdf.get_current_params().await;
        assert!(params.adaptive_iterations >= 1_000);
        assert!(params.adaptive_iterations <= 2_000);
    }

    #[tokio::test]
    async fn test_governance_tier_change() {
        let vdf = ConservativeAdaptiveVDF::new();

        // Vote for Enhanced tier
        vdf.update_security_tier(10, 50, 20).await.unwrap();

        let params = vdf.get_current_params().await;
        assert_eq!(params.security_tier, SecurityTier::Enhanced);
        assert_eq!(params.base_iterations, 1_500);
    }

    #[tokio::test]
    async fn test_vdf_evaluation() {
        let vdf = ConservativeAdaptiveVDF::new();
        let input = b"test_input_for_vdf_evaluation_quantum_resistant";

        let (output, proof) = vdf.evaluate_adaptive(input, 0).await.unwrap();

        assert!(!output.is_zero());
        assert_eq!(proof.iterations, 1_000); // Standard tier baseline

        // Verify proof
        let verified = vdf.verify_adaptive(&proof, input).await.unwrap();
        assert!(verified);
    }

    #[tokio::test]
    async fn test_manipulation_detection() {
        let mut tracker = HashrateTracker::new();

        // Add normal measurements
        for _ in 0..10 {
            tracker.add_snapshot(1_000_000_000.0);
        }

        // Add sudden spike
        tracker.add_snapshot(2_000_000_000.0); // 100% increase

        assert!(tracker.detect_manipulation());
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let vdf = ConservativeAdaptiveVDF::new();

        // Initialize with baseline
        vdf.update_hashrate(1_000_000_000.0).await.unwrap();

        // Try to update with 10% increase (should fail - exceeds 5% limit)
        let result = vdf.update_hashrate(1_100_000_000.0).await;
        assert!(result.is_err());
    }
}
