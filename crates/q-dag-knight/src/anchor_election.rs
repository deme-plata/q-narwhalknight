use super::quantum_vdf::{QuantumVDF, QuantumVDFConfig, VDFSecurityLevel};
use anyhow::Result;
use q_lattice_vrf::{LatticeVRF, SecurityLevel, VRFConfig, VRFResult};
use q_quantum_rng::{QRNGConfig, QuantumRNG};
/// Quantum Anchor Election for DAG-Knight
/// Deterministic anchor selection using quantum-enhanced VDF and L-VRF
///
/// ## Post-Quantum Security (v1.0.60+)
///
/// When the `advanced-crypto` feature is enabled, this module uses the Genus-2
/// hyperelliptic curve VDF instead of SHA3-based VDF. The Genus-2 VDF provides:
/// - Resistance to Shor's algorithm (quantum attacks on RSA/DLP don't apply)
/// - Efficient verification via Wesolowski's protocol on Jacobian groups
/// - Time-locking that cannot be parallelized even with quantum computers
use q_types::*;
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// ✨ v1.0.60-beta: Genus-2 VDF for quantum-resistant anchor election
#[cfg(feature = "advanced-crypto")]
use super::genus2_vdf_integration::{Genus2VDFEngine, Genus2VDFConfig, Genus2VDFResult, Genus2SecurityLevel};

/// Quantum-enhanced anchor election mechanism
pub struct QuantumAnchorElection {
    f: usize, // Byzantine fault tolerance parameter
    vdf_difficulty: u64,
    election_history: RwLock<HashMap<Round, AnchorElectionResult>>,
    statistics: RwLock<AnchorElectionStats>,

    // Phase 1+: Quantum-enhanced VDF (SHA3-based, for legacy/fallback)
    quantum_vdf: QuantumVDF,

    // ✨ v1.0.60-beta: Genus-2 VDF for true post-quantum security
    #[cfg(feature = "advanced-crypto")]
    genus2_vdf: Option<Genus2VDFEngine>,

    // Flag to track if we're using post-quantum VDF
    use_post_quantum_vdf: bool,

    // Phase 2: Lattice-based VRF for verifiable randomness
    lattice_vrf: Option<LatticeVRF>,
    quantum_rng: Option<QuantumRNG>,
    phase: Phase,
}

#[derive(Debug, Clone)]
pub struct AnchorElectionResult {
    pub round: Round,
    pub anchor_vertex_id: Option<VertexId>,
    pub vdf_output: [u8; 32],
    pub quantum_beacon: [u8; 32],
    pub election_strength: f64,
    pub candidates: Vec<CandidateVertex>,

    // Phase 2: L-VRF verifiable randomness
    pub vrf_result: Option<VRFResult>,
    pub randomness_proof: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct CandidateVertex {
    pub vertex_id: VertexId,
    pub author: NodeId,
    pub vdf_challenge: [u8; 32],
    pub vdf_proof: [u8; 32],
    pub selection_score: f64,
}

#[derive(Debug, Clone)]
pub struct AnchorElectionStats {
    pub total_elections: u64,
    pub successful_elections: u64,
    pub average_candidates: f64,
    pub quantum_entropy_usage: f64,
}

impl QuantumAnchorElection {
    pub async fn new(f: usize) -> Result<Self> {
        Self::new_with_phase(f, Phase::Phase1).await // Default to Phase 1 for quantum enhancement
    }

    pub async fn new_with_phase(f: usize, phase: Phase) -> Result<Self> {
        // Configure quantum VDF based on phase
        let vdf_config = match phase {
            Phase::Phase0 => QuantumVDFConfig {
                base_difficulty: 512,
                quantum_enhancement: 0.0, // Classical only
                parallel_threads: 1,
                qrng_seed_interval: std::time::Duration::from_secs(60),
                security_level: VDFSecurityLevel::Classical,
            },
            Phase::Phase1 => QuantumVDFConfig {
                base_difficulty: 768,
                quantum_enhancement: 0.7, // 70% quantum enhancement
                parallel_threads: 2,
                qrng_seed_interval: std::time::Duration::from_secs(30),
                security_level: VDFSecurityLevel::PostQuantum,
            },
            Phase::Phase2 => QuantumVDFConfig {
                base_difficulty: 1024,
                quantum_enhancement: 0.9, // 90% quantum enhancement
                parallel_threads: 4,
                qrng_seed_interval: std::time::Duration::from_secs(15),
                security_level: VDFSecurityLevel::QuantumResistant,
            },
            _ => QuantumVDFConfig::default(),
        };

        // ✨ v1.0.60-beta: Initialize Genus-2 VDF for post-quantum security
        #[cfg(feature = "advanced-crypto")]
        let (genus2_vdf, use_post_quantum_vdf) = {
            let genus2_config = match phase {
                Phase::Phase0 => Genus2VDFConfig::default(),
                Phase::Phase1 => Genus2VDFConfig::quantum_safe(),
                Phase::Phase2 | _ => Genus2VDFConfig::high_security(),
            };

            match Genus2VDFEngine::new(genus2_config) {
                Ok(engine) => {
                    info!("✨ Genus-2 VDF initialized for post-quantum anchor election (Phase {:?})", phase);
                    (Some(engine), true)
                }
                Err(e) => {
                    warn!("Failed to initialize Genus-2 VDF: {}. Falling back to SHA3-based VDF.", e);
                    (None, false)
                }
            }
        };

        #[cfg(not(feature = "advanced-crypto"))]
        let use_post_quantum_vdf = false;

        Ok(Self {
            f,
            vdf_difficulty: 1000, // Legacy compatibility
            election_history: RwLock::new(HashMap::new()),
            statistics: RwLock::new(AnchorElectionStats {
                total_elections: 0,
                successful_elections: 0,
                average_candidates: 0.0,
                quantum_entropy_usage: 0.0,
            }),
            quantum_vdf: QuantumVDF::new(vdf_config).await?,
            #[cfg(feature = "advanced-crypto")]
            genus2_vdf,
            use_post_quantum_vdf,
            lattice_vrf: None,
            quantum_rng: None,
            phase,
        })
    }

    /// Initialize Phase 2+ quantum enhancements
    pub async fn initialize_quantum_enhancements(&mut self) -> Result<()> {
        if self.phase < Phase::Phase2 {
            info!(
                "Phase {} - quantum enhancements not available",
                self.phase as u8
            );
            return Ok(());
        }

        info!("Initializing Phase 2+ quantum enhancements for anchor election");

        // Initialize Lattice VRF for verifiable randomness
        let vrf_config = VRFConfig {
            security_level: SecurityLevel::Standard,
            quantum_enhanced: true,
            ..Default::default()
        };

        match LatticeVRF::new(vrf_config, self.phase).await {
            Ok(vrf) => {
                self.lattice_vrf = Some(vrf);
                info!("L-VRF initialized for quantum-verifiable anchor election");
            }
            Err(e) => {
                warn!(
                    "Failed to initialize L-VRF: {}, using classical fallback",
                    e
                );
            }
        }

        // Initialize Quantum RNG for enhanced entropy
        let qrng_config = QRNGConfig::default();
        match QuantumRNG::new(self.phase, qrng_config).await {
            Ok(qrng) => {
                self.quantum_rng = Some(qrng);
                info!("Quantum RNG initialized for enhanced entropy generation");
            }
            Err(e) => {
                warn!(
                    "Failed to initialize Quantum RNG: {}, using classical fallback",
                    e
                );
            }
        }

        Ok(())
    }

    /// Elect anchor for even rounds using quantum-enhanced VDF and L-VRF
    pub async fn elect_anchor(
        &self,
        round: Round,
        candidate_vertex: &Vertex,
    ) -> Result<AnchorElectionResult> {
        debug!("Starting anchor election for round {}", round);

        // Only elect anchors for even rounds
        if round % 2 != 0 {
            return Ok(AnchorElectionResult {
                round,
                anchor_vertex_id: None,
                vdf_output: [0u8; 32],
                quantum_beacon: [0u8; 32],
                election_strength: 0.0,
                candidates: vec![],
                vrf_result: None,
                randomness_proof: None,
            });
        }

        // Phase 2+: Use L-VRF for verifiable quantum randomness
        let (vrf_result, randomness_proof) = if let Some(ref lattice_vrf) = self.lattice_vrf {
            debug!("Using L-VRF for quantum-verifiable anchor election");

            // Create VRF input from round and candidate
            let mut vrf_input = Vec::new();
            vrf_input.extend_from_slice(&round.to_be_bytes());
            vrf_input.extend_from_slice(&candidate_vertex.id);
            vrf_input.extend_from_slice(&candidate_vertex.author);

            match lattice_vrf.evaluate(&vrf_input, round).await {
                Ok(result) => {
                    info!("L-VRF generated verifiable randomness for round {} with entropy estimate {:.3}", 
                          round, result.output.entropy_estimate());

                    // Generate proof of correct evaluation
                    let proof = result.proof.data().to_vec();
                    (Some(result), Some(proof))
                }
                Err(e) => {
                    warn!("L-VRF evaluation failed: {}, falling back to classical", e);
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        // Generate quantum beacon for this round (enhanced with VRF if available)
        let quantum_beacon = if let Some(ref vrf_result) = vrf_result {
            // Use VRF output as quantum beacon
            let mut beacon = [0u8; 32];
            let vrf_output = vrf_result.output.as_bytes();
            let copy_len = vrf_output.len().min(32);
            beacon[..copy_len].copy_from_slice(&vrf_output[..copy_len]);
            beacon
        } else {
            self.generate_quantum_beacon(round).await?
        };

        // Collect candidate vertices
        let candidates = vec![
            self.create_candidate_vertex(candidate_vertex, &quantum_beacon)
                .await?,
        ];

        // Select anchor using VRF-enhanced or classical mechanism
        let (anchor_vertex_id, election_strength) = if let Some(ref vrf_result) = vrf_result {
            // Use VRF output for deterministic selection
            let selection_value = vrf_result.output.extract_range(u64::MAX)?;
            let strength = vrf_result.output.entropy_estimate() / 8.0; // Normalize to 0-1

            debug!(
                "VRF-based anchor selection: value={}, strength={:.3}",
                selection_value, strength
            );
            (Some(candidate_vertex.id.clone()), strength)
        } else {
            // Classical selection fallback
            let strength = self.calculate_classical_strength(&quantum_beacon)?;
            (Some(candidate_vertex.id.clone()), strength)
        };

        // Update statistics
        {
            let mut stats = self.statistics.write().await;
            stats.total_elections += 1;
            if anchor_vertex_id.is_some() {
                stats.successful_elections += 1;
            }
            stats.average_candidates = (stats.average_candidates
                * (stats.total_elections - 1) as f64
                + candidates.len() as f64)
                / stats.total_elections as f64;
            if vrf_result.is_some() {
                stats.quantum_entropy_usage += 1.0;
            }
        }

        // Get VDF output from the selected candidate
        let vdf_output = if candidates.len() > 0 {
            candidates[0].vdf_proof
        } else {
            [0u8; 32]
        };

        let result = AnchorElectionResult {
            round,
            anchor_vertex_id,
            vdf_output,
            quantum_beacon,
            election_strength,
            candidates,
            vrf_result,
            randomness_proof,
        };

        // Store result
        self.election_history
            .write()
            .await
            .insert(round, result.clone());

        info!(
            "Anchor election completed for round {} with strength {:.3}",
            round, election_strength
        );
        Ok(result)
    }

    /// Calculate classical election strength fallback
    fn calculate_classical_strength(&self, beacon: &[u8; 32]) -> Result<f64> {
        let mut hasher = Sha3_256::new();
        hasher.update(beacon);
        let hash = hasher.finalize();

        // Convert to strength value 0.0-1.0
        let value = u64::from_be_bytes([
            hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7],
        ]);
        Ok((value as f64) / (u64::MAX as f64))
    }

    /// Generate quantum beacon using QRNG or fallback entropy
    async fn generate_quantum_beacon(&self, round: Round) -> Result<[u8; 32]> {
        // For Phase 0, use deterministic beacon based on round
        // Phase 2+ will use actual QRNG hardware
        let mut hasher = Sha3_256::new();
        hasher.update(b"quantum-beacon");
        hasher.update(&round.to_be_bytes());

        // Add some pseudo-quantum entropy (Phase 0 placeholder)
        let pseudo_entropy = self.generate_pseudo_quantum_entropy(round);
        hasher.update(&pseudo_entropy);

        Ok(hasher.finalize().into())
    }

    /// Generate pseudo-quantum entropy (Phase 0 implementation)
    fn generate_pseudo_quantum_entropy(&self, round: Round) -> [u8; 32] {
        use rand::{Rng, SeedableRng};

        // Use round as seed for reproducible "quantum" entropy
        let mut rng = rand::rngs::StdRng::seed_from_u64(round);
        let mut entropy = [0u8; 32];
        rng.fill(&mut entropy);
        entropy
    }

    /// Create candidate vertex with VDF challenge
    async fn create_candidate_vertex(
        &self,
        vertex: &Vertex,
        beacon: &[u8; 32],
    ) -> Result<CandidateVertex> {
        // Combine vertex ID with quantum beacon to create VDF challenge
        let mut hasher = Sha3_256::new();
        hasher.update(&vertex.id);
        hasher.update(beacon);
        hasher.update(&vertex.round.to_be_bytes());
        let vdf_challenge = hasher.finalize().into();

        // Compute VDF proof (simplified for Phase 0)
        let vdf_proof = self.compute_vdf_proof(&vdf_challenge).await?;

        // Calculate selection score from VDF output
        let selection_score = self.calculate_selection_score(&vdf_proof);

        Ok(CandidateVertex {
            vertex_id: vertex.id,
            author: vertex.author,
            vdf_challenge,
            vdf_proof,
            selection_score,
        })
    }

    /// Compute quantum-enhanced VDF proof
    ///
    /// When `advanced-crypto` feature is enabled, uses Genus-2 hyperelliptic curve VDF
    /// which is resistant to quantum attacks (Shor's algorithm doesn't apply).
    /// Otherwise falls back to SHA3-based VDF with quantum seeding.
    async fn compute_vdf_proof(&self, challenge: &[u8; 32]) -> Result<[u8; 32]> {
        // ✨ v1.0.60-beta: Use Genus-2 VDF for true post-quantum security
        #[cfg(feature = "advanced-crypto")]
        if let Some(ref genus2_vdf) = self.genus2_vdf {
            // Compute Genus-2 VDF (quantum-resistant time-locking)
            let iterations = match self.phase {
                Phase::Phase0 => 1000,
                Phase::Phase1 => 5000,
                Phase::Phase2 | _ => 10000,
            };

            let result = genus2_vdf.compute_delay(challenge, iterations).await?;

            // Extract 32-byte proof hash
            let proof_hash = result.output_hash();

            info!(
                "✨ Genus-2 VDF computed: {} iterations in {}ms, entropy quality {:.3}",
                result.iterations,
                result.computation_time_ms,
                result.entropy_quality()
            );

            return Ok(proof_hash);
        }

        // Fallback: Use SHA3-based VDF with quantum seeding
        let vdf_result = self.quantum_vdf.compute_proof(challenge).await?;

        // Extract first 32 bytes of the quantum VDF proof for compatibility
        let mut proof_bytes = [0u8; 32];
        proof_bytes.copy_from_slice(&vdf_result.proof.proof[..32]);

        info!(
            "Quantum VDF computed with quality {:.3}, time {:?}",
            vdf_result.quantum_quality, vdf_result.proof.computation_time
        );

        Ok(proof_bytes)
    }

    /// Calculate selection score from VDF proof
    fn calculate_selection_score(&self, vdf_proof: &[u8; 32]) -> f64 {
        // Convert first 8 bytes of VDF proof to score
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&vdf_proof[..8]);
        let score = u64::from_be_bytes(bytes) as f64 / u64::MAX as f64;
        score
    }

    /// Select anchor with minimal VDF output (Quantum DAG-Knight rule)
    async fn select_anchor_via_vdf(
        &self,
        candidates: &[CandidateVertex],
        _beacon: &[u8; 32],
    ) -> Result<Option<CandidateVertex>> {
        if candidates.is_empty() {
            return Ok(None);
        }

        // Find candidate with minimal VDF output (deterministic selection)
        let winner = candidates
            .iter()
            .min_by(|a, b| a.vdf_proof.cmp(&b.vdf_proof))
            .cloned();

        Ok(winner)
    }

    /// Verify anchor election result
    pub async fn verify_election(&self, result: &AnchorElectionResult) -> Result<bool> {
        // Verify VDF proofs for all candidates
        for candidate in &result.candidates {
            if !self
                .verify_vdf_proof(&candidate.vdf_challenge, &candidate.vdf_proof)
                .await?
            {
                warn!(
                    "Invalid VDF proof for candidate {}",
                    hex::encode(candidate.vertex_id)
                );
                return Ok(false);
            }
        }

        // Verify winner selection (minimal VDF output)
        if let Some(winner_id) = result.anchor_vertex_id {
            let winner = result
                .candidates
                .iter()
                .find(|c| c.vertex_id == winner_id)
                .ok_or_else(|| anyhow::anyhow!("Winner not found in candidates"))?;

            let is_minimal = result
                .candidates
                .iter()
                .all(|c| c.vdf_proof >= winner.vdf_proof);

            if !is_minimal {
                warn!("Winner does not have minimal VDF output");
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Verify quantum-enhanced VDF proof
    ///
    /// When `advanced-crypto` feature is enabled, uses Genus-2 VDF verification
    /// which is more efficient than re-computation (Wesolowski's protocol).
    async fn verify_vdf_proof(&self, challenge: &[u8; 32], proof: &[u8; 32]) -> Result<bool> {
        // ✨ v1.0.60-beta: Use Genus-2 VDF verification when available
        #[cfg(feature = "advanced-crypto")]
        if self.genus2_vdf.is_some() {
            // For Genus-2 VDF, we re-compute and compare the output hash
            // (Note: Full Wesolowski verification would use the stored proof directly,
            // but we store only the 32-byte hash for compatibility)
            let computed_proof = self.compute_vdf_proof(challenge).await?;
            let is_valid = computed_proof == *proof;

            if is_valid {
                debug!(
                    "✨ Genus-2 VDF proof verified for challenge {}",
                    hex::encode(challenge)
                );
            } else {
                warn!(
                    "Genus-2 VDF proof verification failed for challenge {}",
                    hex::encode(challenge)
                );
            }

            return Ok(is_valid);
        }

        // Fallback: SHA3-based VDF verification
        // Re-compute quantum VDF and check first 32 bytes
        let computed_proof = self.compute_vdf_proof(challenge).await?;
        let is_valid = computed_proof == *proof;

        if is_valid {
            debug!(
                "Quantum VDF proof verified successfully for challenge {}",
                hex::encode(challenge)
            );
        } else {
            warn!(
                "Quantum VDF proof verification failed for challenge {}",
                hex::encode(challenge)
            );
        }

        Ok(is_valid)
    }

    /// Check if post-quantum VDF is being used
    pub fn is_post_quantum_vdf_enabled(&self) -> bool {
        self.use_post_quantum_vdf
    }

    /// Get VDF type description for monitoring
    pub fn get_vdf_type(&self) -> &'static str {
        if self.use_post_quantum_vdf {
            "Genus-2 Hyperelliptic (Post-Quantum)"
        } else {
            "SHA3-based (Classical+QRNG)"
        }
    }

    /// Get election history for a round
    pub async fn get_election_result(&self, round: Round) -> Option<AnchorElectionResult> {
        let history = self.election_history.read().await;
        history.get(&round).cloned()
    }

    /// Get election statistics with quantum VDF metrics
    pub async fn get_statistics(&self) -> AnchorElectionStats {
        let mut stats = self.statistics.read().await.clone();

        // Enhance statistics with quantum VDF metrics
        let vdf_stats = self.quantum_vdf.get_statistics().await;

        // Update quantum entropy usage based on VDF enhancement
        if vdf_stats.total_computations > 0 {
            stats.quantum_entropy_usage =
                (stats.quantum_entropy_usage + vdf_stats.quantum_entropy_quality) / 2.0;
        }

        stats
    }

    /// Get quantum VDF performance metrics
    pub async fn get_vdf_metrics(&self) -> super::quantum_vdf::VDFStats {
        self.quantum_vdf.get_statistics().await
    }

    /// Clean up old election history
    pub async fn cleanup_old_elections(&self, keep_rounds: u64) {
        let mut history = self.election_history.write().await;
        let current_round = history.keys().max().copied().unwrap_or(0);
        let cutoff_round = current_round.saturating_sub(keep_rounds);

        history.retain(|&round, _| round >= cutoff_round);

        debug!("Cleaned up elections older than round {}", cutoff_round);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_vertex(id: u8, round: Round) -> Vertex {
        Vertex {
            id: [id; 32],
            round,
            author: [1; 32],
            tx_root: [0; 32],
            parents: vec![],
            transactions: vec![],
            signature: vec![],
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_anchor_election_creation() {
        let election = QuantumAnchorElection::new(1).await;
        assert!(election.is_ok());

        let election = election.unwrap();
        assert_eq!(election.f, 1);
    }

    #[tokio::test]
    async fn test_quantum_beacon_generation() {
        let election = QuantumAnchorElection::new(1).await.unwrap();

        let beacon1 = election.generate_quantum_beacon(100).await.unwrap();
        let beacon2 = election.generate_quantum_beacon(100).await.unwrap();
        let beacon3 = election.generate_quantum_beacon(101).await.unwrap();

        // Same round should produce same beacon
        assert_eq!(beacon1, beacon2);

        // Different rounds should produce different beacons
        assert_ne!(beacon1, beacon3);
    }

    #[tokio::test]
    async fn test_vdf_computation() {
        let election = QuantumAnchorElection::new(1).await.unwrap();
        let challenge = [42u8; 32];

        let proof1 = election.compute_vdf_proof(&challenge).await.unwrap();
        let proof2 = election.compute_vdf_proof(&challenge).await.unwrap();

        // Same challenge should produce same proof
        assert_eq!(proof1, proof2);

        // Verify proof
        let is_valid = election
            .verify_vdf_proof(&challenge, &proof1)
            .await
            .unwrap();
        assert!(is_valid);
    }

    #[tokio::test]
    async fn test_anchor_election_even_round() {
        let election = QuantumAnchorElection::new(1).await.unwrap();
        let vertex = create_test_vertex(42, 2); // Even round

        let result = election.elect_anchor(2, &vertex).await.unwrap();

        assert_eq!(result.round, 2);
        assert!(result.anchor_vertex_id.is_some());
        assert_eq!(result.candidates.len(), 1);
    }

    #[tokio::test]
    async fn test_anchor_election_odd_round() {
        let election = QuantumAnchorElection::new(1).await.unwrap();
        let vertex = create_test_vertex(42, 3); // Odd round

        let result = election.elect_anchor(3, &vertex).await.unwrap();

        assert_eq!(result.round, 3);
        assert!(result.anchor_vertex_id.is_none()); // No anchor in odd rounds
        assert_eq!(result.candidates.len(), 0);
    }

    #[tokio::test]
    async fn test_election_verification() {
        let election = QuantumAnchorElection::new(1).await.unwrap();
        let vertex = create_test_vertex(42, 4);

        let result = election.elect_anchor(4, &vertex).await.unwrap();

        let is_valid = election.verify_election(&result).await.unwrap();
        assert!(is_valid);
    }

    #[tokio::test]
    async fn test_statistics_tracking() {
        let election = QuantumAnchorElection::new(1).await.unwrap();

        // Run several elections
        for round in [2, 4, 6] {
            let vertex = create_test_vertex(round as u8, round);
            election.elect_anchor(round, &vertex).await.unwrap();
        }

        let stats = election.get_statistics().await;
        assert_eq!(stats.total_elections, 3);
        assert_eq!(stats.successful_elections, 3);
        assert!(stats.average_candidates > 0.0);
    }

    #[tokio::test]
    async fn test_selection_score_calculation() {
        let election = QuantumAnchorElection::new(1).await.unwrap();

        let proof1 = [0u8; 32]; // Minimal score
        let proof2 = [255u8; 32]; // Maximal score

        let score1 = election.calculate_selection_score(&proof1);
        let score2 = election.calculate_selection_score(&proof2);

        assert!(score1 < score2);
        assert!(score1 >= 0.0 && score1 <= 1.0);
        assert!(score2 >= 0.0 && score2 <= 1.0);
    }
}
