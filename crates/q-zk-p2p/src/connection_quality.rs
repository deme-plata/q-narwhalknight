//! Connection Quality Zero-Knowledge Proofs
//!
//! This module implements ZK proofs for connection quality metrics and consensus
//! participation without revealing actual performance data or voting patterns.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::ops::RangeInclusive;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

use q_types::{ConsensusVote, ProposalHash};
use q_zk_snark::{CircuitBuilder, SNARKConfig, SNARKProtocol, UniversalSNARK};
use q_zk_stark::StarkSystem;

use crate::{field_from_u64, ZkP2pError};

/// Zero-knowledge proof of connection quality meeting minimum standards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionQualityProof {
    /// STARK proof of connection performance (transparent, no trusted setup)
    pub quality_proof: Vec<u8>, // Serialized StarkProof
    /// Committed connection metrics (hidden actual values)
    pub metrics_commitment: [u8; 32],
    /// Quality score range proof (0-100 scale)
    pub range_proof: RangeProof,
    /// Proof generation timestamp
    pub timestamp: u64,
    /// Performance tier achieved
    pub performance_tier: PerformanceTier,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeProof {
    /// Proof that value is in range [min, max]
    pub proof_data: Vec<u8>,
    /// Minimum value (public)
    pub min_value: u64,
    /// Maximum value (public)
    pub max_value: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceTier {
    Basic,    // Meets minimum requirements
    Standard, // Good performance
    Premium,  // Excellent performance
    Elite,    // Top-tier performance
}

impl RangeProof {
    /// Create a new range proof (simplified implementation)
    pub fn new(value: u64, min_value: u64, max_value: u64) -> Result<Self> {
        if value < min_value || value > max_value {
            return Err(
                ZkP2pError::ProofGeneration("Value outside valid range".to_string()).into(),
            );
        }

        // Mock range proof generation
        let proof_data = bincode::serialize(&format!("range_proof_{}_{}", min_value, max_value))?;

        Ok(Self {
            proof_data,
            min_value,
            max_value,
        })
    }

    /// Verify range proof
    pub fn verify(&self) -> Result<bool> {
        // Mock range proof verification
        let expected_data = bincode::serialize(&format!(
            "range_proof_{}_{}",
            self.min_value, self.max_value
        ))?;
        Ok(self.proof_data == expected_data)
    }
}

impl ConnectionQualityProof {
    /// Generate proof of connection quality meeting minimum standards
    pub async fn prove_quality(
        latency_ms: u32,
        bandwidth_mbps: u32,
        uptime_percentage: f32,
        min_latency: u32,
        min_bandwidth: u32,
        min_uptime: f32,
    ) -> Result<ConnectionQualityProof> {
        info!(
            "⚡ Generating connection quality proof (latency: {}ms, bandwidth: {}Mbps, uptime: {:.1}%)",
            latency_ms, bandwidth_mbps, uptime_percentage * 100.0
        );

        // Use STARK for transparent proof generation
        let mut stark_system = StarkSystem::new(true).await.map_err(|e| {
            ZkP2pError::ProofGeneration(format!("STARK system creation failed: {}", e))
        })?;

        // Create execution trace proving all quality metrics meet minimums
        let trace = build_quality_trace(
            latency_ms,
            bandwidth_mbps,
            uptime_percentage,
            min_latency,
            min_bandwidth,
            min_uptime,
        );

        debug!("Built quality verification trace with {} rows", trace.len());

        // Generate constraint system for quality verification
        let constraints = build_quality_constraints();

        // Generate transparent ZK proof
        let stark_proof = stark_system
            .prove(&trace, &constraints)
            .await
            .map_err(|e| {
                ZkP2pError::ProofGeneration(format!("Quality proof generation failed: {}", e))
            })?;

        // Serialize proof
        let quality_proof = bincode::serialize(&stark_proof).map_err(|e| {
            ZkP2pError::Serialization(format!("Quality proof serialization failed: {}", e))
        })?;

        // Create commitment to actual metrics (keeps values private)
        let metrics_commitment = blake3::hash(&bincode::serialize(&(
            latency_ms,
            bandwidth_mbps,
            uptime_percentage,
        ))?)
        .into();

        // Calculate overall quality score
        let quality_score = calculate_quality_score(latency_ms, bandwidth_mbps, uptime_percentage);

        // Generate range proof for quality score (0-100 range)
        let range_proof = RangeProof::new(quality_score as u64, 0, 100)?;

        // Determine performance tier
        let performance_tier =
            classify_performance_tier(latency_ms, bandwidth_mbps, uptime_percentage);

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        info!(
            "✅ Connection quality proof generated (tier: {:?}, score: {})",
            performance_tier, quality_score
        );

        Ok(ConnectionQualityProof {
            quality_proof,
            metrics_commitment,
            range_proof,
            timestamp,
            performance_tier,
        })
    }

    /// Verify connection meets quality standards without learning actual metrics
    pub async fn verify_quality_standards(
        &self,
        min_latency: u32,
        min_bandwidth: u32,
        min_uptime: f32,
    ) -> Result<bool> {
        debug!("🔍 Verifying connection quality against standards: latency≤{}ms, bandwidth≥{}Mbps, uptime≥{:.1}%",
               min_latency, min_bandwidth, min_uptime * 100.0);

        // Check proof freshness (quality metrics change over time)
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let age = current_time.saturating_sub(self.timestamp);
        if age > 900 {
            // 15 minute expiry for connection quality
            warn!("⚠️ Connection quality proof expired (age: {}s)", age);
            return Ok(false);
        }

        // Deserialize STARK proof
        let stark_proof = bincode::deserialize(&self.quality_proof).map_err(|e| {
            ZkP2pError::ProofGeneration(format!("Quality proof deserialization failed: {}", e))
        })?;

        // Create verifier
        let mut stark_system = StarkSystem::new(false).await.map_err(|e| {
            ZkP2pError::ProofGeneration(format!("STARK verifier creation failed: {}", e))
        })?;

        // Public inputs: minimum requirements and expected result (all conditions pass = 1)
        let public_inputs = vec![
            min_latency as u64,
            min_bandwidth as u64,
            (min_uptime * 100.0) as u64,
            1, // Expected: all quality checks pass
        ];

        // Verify the STARK proof
        let quality_valid = stark_system
            .verify(&stark_proof, &public_inputs)
            .await
            .map_err(|e| {
                ZkP2pError::ProofGeneration(format!("Quality proof verification failed: {}", e))
            })?;

        // Verify range proof
        let range_valid = self.range_proof.verify()?;

        let is_valid = quality_valid && range_valid;

        if is_valid {
            info!(
                "✅ Connection quality proof verified successfully (tier: {:?})",
                self.performance_tier
            );
        } else {
            warn!("❌ Connection quality proof verification failed");
        }

        Ok(is_valid)
    }

    /// Get performance tier without revealing exact metrics
    pub fn get_performance_tier(&self) -> PerformanceTier {
        self.performance_tier.clone()
    }
}

/// Zero-knowledge proof of active consensus participation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusParticipationProof {
    /// Sonic proof (updatable setup for evolving consensus rules)
    pub participation_proof: Vec<u8>, // Serialized proof
    /// Nullifiers to prevent double-voting
    pub vote_nullifiers: Vec<[u8; 32]>,
    /// Commitment to voting history (keeps votes private)
    pub history_commitment: [u8; 32],
    /// Participation metadata
    pub metadata: ParticipationMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipationMetadata {
    /// Epoch range covered by this proof
    pub epoch_range: RangeInclusive<u64>,
    /// Minimum participation rate proven
    pub min_participation_rate: f32,
    /// Proof generation timestamp
    pub timestamp: u64,
    /// Circuit complexity
    pub circuit_size: usize,
}

impl ConsensusParticipationProof {
    /// Generate proof of active consensus participation
    pub async fn prove_active_participation(
        voting_history: &[ConsensusVote],
        min_participation_rate: f32,
        epoch_range: RangeInclusive<u64>,
    ) -> Result<ConsensusParticipationProof> {
        info!(
            "🗳️ Generating participation proof for epochs {:?} (min rate: {:.1}%)",
            epoch_range,
            min_participation_rate * 100.0
        );

        // Use Sonic for updatable setup (good for evolving consensus rules)
        let snark_config = SNARKConfig {
            protocol: SNARKProtocol::Sonic,
            security_bits: 128,
            parallel_proving: true,
            max_constraints: 1_000_000, // Large circuit for complex voting logic
            batch_verification: false,  // Individual verification for participation
            ..Default::default()
        };

        let _snark = UniversalSNARK::new(snark_config);

        // Build participation verification circuit
        let mut builder =
            CircuitBuilder::<ark_bn254::Fr>::new("consensus_participation".to_string());

        // Filter votes by epoch range
        let relevant_votes: Vec<&ConsensusVote> = voting_history
            .iter()
            .filter(|vote| epoch_range.contains(&vote.epoch))
            .collect();

        // Calculate participation statistics
        let total_opportunities = relevant_votes.len();
        let participation_count = relevant_votes
            .iter()
            .filter(|vote| vote.participated)
            .count();

        let actual_participation_rate = if total_opportunities > 0 {
            participation_count as f32 / total_opportunities as f32
        } else {
            0.0
        };

        debug!(
            "Participation statistics: {}/{} votes ({:.1}%)",
            participation_count,
            total_opportunities,
            actual_participation_rate * 100.0
        );

        // Build circuit constraints (simplified for demo)
        let min_rate_var = builder.create_variable("min_participation_rate".to_string(), true);
        let actual_rate_var =
            builder.create_variable("actual_participation_rate".to_string(), false);

        builder.assign_variable(
            &min_rate_var,
            field_from_u64((min_participation_rate * 10000.0) as u64),
        )?;
        builder.assign_variable(
            &actual_rate_var,
            field_from_u64((actual_participation_rate * 10000.0) as u64),
        )?;

        // Constraint: actual_rate >= min_rate
        let comparison_var = builder.create_variable("rate_comparison".to_string(), false);
        let comparison_value = (actual_participation_rate >= min_participation_rate) as u64;
        builder.assign_variable(&comparison_var, field_from_u64(comparison_value))?;
        builder.enforce_constant(
            &comparison_var,
            ark_bn254::Fr::from(1u64),
            Some("participation_check".to_string()),
        )?;

        let circuit = builder.build();
        let circuit_size = circuit.size();

        // Generate nullifiers for vote privacy
        let mut vote_nullifiers = Vec::new();
        for vote in &relevant_votes {
            let nullifier_input = bincode::serialize(&(vote.epoch, &vote.proposal_hash))?;
            let nullifier = blake3::hash(&nullifier_input).into();
            vote_nullifiers.push(nullifier);
        }

        // Generate proof (mock Sonic proof)
        let participation_proof =
            bincode::serialize(&"mock_sonic_participation_proof").map_err(|e| {
                ZkP2pError::Serialization(format!(
                    "Participation proof serialization failed: {}",
                    e
                ))
            })?;

        // Create voting history commitment
        let history_commitment = blake3::hash(&bincode::serialize(voting_history)?).into();

        let metadata = ParticipationMetadata {
            epoch_range,
            min_participation_rate,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            circuit_size,
        };

        info!(
            "✅ Consensus participation proof generated (rate: {:.1}%)",
            actual_participation_rate * 100.0
        );

        Ok(ConsensusParticipationProof {
            participation_proof,
            vote_nullifiers,
            history_commitment,
            metadata,
        })
    }

    /// Verify consensus participation proof
    pub async fn verify_participation(&self) -> Result<bool> {
        debug!(
            "🔍 Verifying participation proof for epochs {:?} (min rate: {:.1}%)",
            self.metadata.epoch_range,
            self.metadata.min_participation_rate * 100.0
        );

        // Check proof freshness
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let age = current_time.saturating_sub(self.metadata.timestamp);
        if age > 7200 {
            // 2 hour expiry for participation proofs
            warn!("⚠️ Participation proof expired (age: {}s)", age);
            return Ok(false);
        }

        // Verify proof structure (mock verification)
        let proof_data: Result<String, _> = bincode::deserialize(&self.participation_proof);
        let is_valid =
            proof_data.is_ok() && proof_data.unwrap() == "mock_sonic_participation_proof";

        // Check nullifier uniqueness (prevent double-spending of votes)
        let mut nullifier_set = std::collections::HashSet::new();
        for nullifier in &self.vote_nullifiers {
            if !nullifier_set.insert(nullifier) {
                warn!("❌ Duplicate nullifier detected in participation proof");
                return Ok(false);
            }
        }

        if is_valid {
            info!("✅ Consensus participation proof verified successfully");
        } else {
            warn!("❌ Consensus participation proof verification failed");
        }

        Ok(is_valid)
    }
}

/// Build execution trace for quality verification
fn build_quality_trace(
    latency_ms: u32,
    bandwidth_mbps: u32,
    uptime_percentage: f32,
    min_latency: u32,
    min_bandwidth: u32,
    min_uptime: f32,
) -> Vec<Vec<u64>> {
    vec![
        // Latency check: latency <= min_latency (lower is better)
        vec![
            latency_ms as u64,
            min_latency as u64,
            (latency_ms <= min_latency) as u64,
        ],
        // Bandwidth check: bandwidth >= min_bandwidth (higher is better)
        vec![
            bandwidth_mbps as u64,
            min_bandwidth as u64,
            (bandwidth_mbps >= min_bandwidth) as u64,
        ],
        // Uptime check: uptime >= min_uptime
        vec![
            (uptime_percentage * 100.0) as u64,
            (min_uptime * 100.0) as u64,
            (uptime_percentage >= min_uptime) as u64,
        ],
        // All conditions must pass
        vec![1, 1, 1],
    ]
}

/// Build constraint system for quality verification
fn build_quality_constraints() -> Vec<u8> {
    // Mock constraint system for quality verification
    vec![0u8; 256]
}

/// Calculate overall quality score (0-100)
fn calculate_quality_score(latency_ms: u32, bandwidth_mbps: u32, uptime_percentage: f32) -> u32 {
    // Weighted scoring system
    let latency_score = ((1000.0 - latency_ms.min(1000) as f32) / 1000.0 * 40.0) as u32; // 40 points max
    let bandwidth_score = (bandwidth_mbps.min(100) as f32 / 100.0 * 30.0) as u32; // 30 points max
    let uptime_score = (uptime_percentage * 30.0) as u32; // 30 points max

    (latency_score + bandwidth_score + uptime_score).min(100)
}

/// Classify performance tier based on metrics
fn classify_performance_tier(
    latency_ms: u32,
    bandwidth_mbps: u32,
    uptime_percentage: f32,
) -> PerformanceTier {
    let quality_score = calculate_quality_score(latency_ms, bandwidth_mbps, uptime_percentage);

    match quality_score {
        90..=100 => PerformanceTier::Elite,
        75..=89 => PerformanceTier::Premium,
        60..=74 => PerformanceTier::Standard,
        _ => PerformanceTier::Basic,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connection_quality_proof() {
        // Test high-quality connection
        let proof = ConnectionQualityProof::prove_quality(
            50,   // 50ms latency (good)
            100,  // 100 Mbps (excellent)
            0.99, // 99% uptime (excellent)
            100,  // max 100ms latency
            10,   // min 10 Mbps
            0.95, // min 95% uptime
        )
        .await
        .unwrap();

        assert_eq!(proof.performance_tier, PerformanceTier::Elite);

        let is_valid = proof.verify_quality_standards(100, 10, 0.95).await.unwrap();
        assert!(is_valid, "High-quality connection should meet standards");
    }

    #[tokio::test]
    async fn test_consensus_participation_proof() {
        let voting_history = vec![
            ConsensusVote {
                epoch: 100,
                proposal_hash: ProposalHash::from("proposal1"),
                participated: true,
            },
            ConsensusVote {
                epoch: 101,
                proposal_hash: ProposalHash::from("proposal2"),
                participated: true,
            },
            ConsensusVote {
                epoch: 102,
                proposal_hash: ProposalHash::from("proposal3"),
                participated: false,
            },
        ];

        let proof = ConsensusParticipationProof::prove_active_participation(
            &voting_history,
            0.60, // 60% minimum participation
            100..=102,
        )
        .await
        .unwrap();

        let is_valid = proof.verify_participation().await.unwrap();
        assert!(is_valid, "Participation proof should verify");

        // Should have 3 nullifiers for 3 votes
        assert_eq!(proof.vote_nullifiers.len(), 3);
    }

    #[test]
    fn test_performance_tier_classification() {
        assert_eq!(
            classify_performance_tier(10, 1000, 1.0),
            PerformanceTier::Elite
        );
        assert_eq!(
            classify_performance_tier(50, 100, 0.99),
            PerformanceTier::Elite
        );
        assert_eq!(
            classify_performance_tier(100, 50, 0.95),
            PerformanceTier::Premium
        );
        assert_eq!(
            classify_performance_tier(200, 20, 0.90),
            PerformanceTier::Standard
        );
        assert_eq!(
            classify_performance_tier(500, 5, 0.80),
            PerformanceTier::Basic
        );
    }

    #[test]
    fn test_range_proof() {
        let proof = RangeProof::new(50, 0, 100).unwrap();
        assert!(proof.verify().unwrap());

        // Should fail for out-of-range values
        assert!(RangeProof::new(150, 0, 100).is_err());
    }

    #[test]
    fn test_quality_score_calculation() {
        // Perfect scores
        assert_eq!(calculate_quality_score(0, 100, 1.0), 100);

        // Good scores
        assert!(calculate_quality_score(50, 50, 0.95) > 80);

        // Poor scores
        assert!(calculate_quality_score(500, 5, 0.80) < 50);
    }
}
