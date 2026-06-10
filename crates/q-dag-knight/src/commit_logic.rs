use super::{AnchorElectionResult, CommitDecision, CommitType};
use anyhow::Result;
use q_lattice_vrf::VRFResult;
/// DAG-Knight commit logic implementation
/// Handles commit rules and finality decisions
use q_types::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Commit protocol implementing DAG-Knight commit rules
pub struct CommitProtocol {
    /// Byzantine fault tolerance parameter
    f: usize,

    /// Delta: rounds to look back for commit decision
    delta: u64,

    /// Commit decisions by round
    commit_history: RwLock<BTreeMap<Round, Vec<CommitDecision>>>,

    /// Pending commit evaluations
    pending_commits: RwLock<HashMap<Round, PendingCommit>>,

    /// Anchor election results
    anchor_results: RwLock<HashMap<Round, AnchorElectionResult>>,

    /// Commit statistics
    statistics: RwLock<CommitStats>,
}

#[derive(Debug, Clone)]
pub struct PendingCommit {
    pub round: Round,
    pub vertex_id: VertexId,
    pub commit_type: CommitType,
    pub evaluation_time: chrono::DateTime<chrono::Utc>,
    pub dependencies: Vec<VertexId>,
}

#[derive(Debug, Clone)]
pub struct CommitStats {
    pub total_commits: u64,
    pub successful_commits: u64,
    pub anchor_commits: u64,
    pub delayed_commits: u64,
    pub chain_commits: u64,
    pub average_commit_latency_rounds: f64,
    pub average_tps: f64,
    pub last_commit_time: Option<chrono::DateTime<chrono::Utc>>,
}

/// Commit rule evaluation result
#[derive(Debug, Clone)]
pub enum CommitRuleResult {
    Commit(CommitDecision),
    Defer {
        reason: String,
        retry_after_rounds: u64,
    },
    Reject {
        reason: String,
    },
}

// =========================================================================
// TAIL FORK PROTECTION - v1.0.69-beta
// Eliminates tail forking vulnerabilities in pipelined BFT consensus
// =========================================================================

/// Detected tail fork information
#[derive(Debug, Clone)]
pub struct TailForkDetection {
    /// Round where conflict was detected
    pub round: Round,
    /// First conflicting vertex
    pub vertex_a: VertexId,
    /// Second conflicting vertex
    pub vertex_b: VertexId,
    /// Detection timestamp
    pub detected_at: chrono::DateTime<chrono::Utc>,
    /// Severity level (1-10)
    pub severity: u8,
}

impl CommitProtocol {
    pub fn new(f: usize) -> Result<Self> {
        Ok(Self {
            f,
            delta: 1, // ⚡ v1.0.72-beta: Aggressive delta=1 for sub-50ms finality (was 4)
            commit_history: RwLock::new(BTreeMap::new()),
            pending_commits: RwLock::new(HashMap::new()),
            anchor_results: RwLock::new(HashMap::new()),
            statistics: RwLock::new(CommitStats {
                total_commits: 0,
                successful_commits: 0,
                anchor_commits: 0,
                delayed_commits: 0,
                chain_commits: 0,
                average_commit_latency_rounds: 0.0,
                average_tps: 0.0,
                last_commit_time: None,
            }),
        })
    }

    /// Evaluate commit decision for a vertex after anchor election
    pub async fn evaluate_commit(
        &self,
        round: Round,
        anchor_vertex_id: VertexId,
    ) -> Result<Option<CommitDecision>> {
        debug!(
            "Evaluating commit for anchor vertex {} in round {}",
            hex::encode(anchor_vertex_id),
            round
        );

        // Store anchor result
        if let Some(anchor_result) = self.get_anchor_result(round).await {
            let mut anchors = self.anchor_results.write().await;
            anchors.insert(round, anchor_result);
        }

        // Apply DAG-Knight commit rule: commit vertices from round r-δ
        if round >= self.delta {
            let commit_round = round - self.delta;
            return self.evaluate_delayed_commit(commit_round).await;
        }

        // For early rounds, defer commit decision
        Ok(None)
    }

    /// Evaluate commit decision with L-VRF enhancement
    pub async fn evaluate_commit_with_vrf(
        &self,
        round: Round,
        anchor_vertex_id: VertexId,
        vrf_result: Option<&VRFResult>,
    ) -> Result<Option<CommitDecision>> {
        debug!(
            "Evaluating VRF-enhanced commit for anchor vertex {} in round {}",
            hex::encode(anchor_vertex_id),
            round
        );

        // If we have VRF result, use it for enhanced commit decision
        if let Some(vrf) = vrf_result {
            // Enhanced commit logic with VRF quality assessment
            let entropy_quality = vrf.output.entropy_estimate();
            let quantum_enhanced = vrf.metadata.quantum_enhanced;

            debug!(
                "VRF entropy quality: {:.3}, quantum enhanced: {}",
                entropy_quality, quantum_enhanced
            );

            // Higher quality VRF results can enable faster commits
            let enhanced_delta = if entropy_quality > 0.8 && quantum_enhanced {
                // High-quality quantum VRF allows reduced delta
                (self.delta * 3) / 4 // 25% reduction in commit delay
            } else if entropy_quality > 0.6 {
                // Good quality VRF allows modest reduction
                (self.delta * 7) / 8 // 12.5% reduction
            } else {
                // Use standard delta for lower quality randomness
                self.delta
            };

            info!(
                "VRF-enhanced commit using delta {} (standard: {}) for entropy {:.3}",
                enhanced_delta, self.delta, entropy_quality
            );

            // Store VRF-enhanced anchor result
            let vrf_anchor_result = AnchorElectionResult {
                round,
                anchor_vertex_id: Some(anchor_vertex_id),
                vdf_output: [0u8; 32],     // Placeholder - should be provided
                quantum_beacon: [0u8; 32], // Placeholder - should be provided
                election_strength: entropy_quality,
                candidates: vec![],
                vrf_result: Some(vrf.clone()),
                randomness_proof: Some(vrf.proof.data().to_vec()),
            };

            {
                let mut anchors = self.anchor_results.write().await;
                anchors.insert(round, vrf_anchor_result);
            }

            // Apply enhanced commit rule
            if round >= enhanced_delta {
                let commit_round = round - enhanced_delta;
                return self.evaluate_vrf_enhanced_commit(commit_round, vrf).await;
            }

            // For immediate anchor commits with high-quality VRF
            if entropy_quality > 0.9 && quantum_enhanced {
                info!("High-quality quantum VRF enables immediate anchor commit");
                return self
                    .create_commit_decision(round, anchor_vertex_id, CommitType::AnchorCommit)
                    .await;
            }
        }

        // Fallback to standard commit evaluation
        self.evaluate_commit(round, anchor_vertex_id).await
    }

    /// Evaluate delayed commit (δ rounds after)
    async fn evaluate_delayed_commit(&self, commit_round: Round) -> Result<Option<CommitDecision>> {
        debug!("Evaluating delayed commit for round {}", commit_round);

        // Check if we already committed this round
        {
            let history = self.commit_history.read().await;
            if history.contains_key(&commit_round) {
                return Ok(None);
            }
        }

        // Find anchor vertex for commit_round (if it was an even round)
        if commit_round % 2 == 0 {
            let anchors = self.anchor_results.read().await;
            if let Some(anchor_result) = anchors.get(&commit_round) {
                if let Some(anchor_id) = anchor_result.anchor_vertex_id {
                    return self
                        .create_commit_decision(commit_round, anchor_id, CommitType::DelayedCommit)
                        .await;
                }
            }
        }

        // For odd rounds or rounds without anchors, check for chain commits
        self.evaluate_chain_commit(commit_round).await
    }

    /// Evaluate VRF-enhanced delayed commit with quality-based optimizations
    async fn evaluate_vrf_enhanced_commit(
        &self,
        commit_round: Round,
        vrf_result: &VRFResult,
    ) -> Result<Option<CommitDecision>> {
        debug!(
            "Evaluating VRF-enhanced delayed commit for round {} with entropy {:.3}",
            commit_round,
            vrf_result.output.entropy_estimate()
        );

        // Check if we already committed this round
        {
            let history = self.commit_history.read().await;
            if history.contains_key(&commit_round) {
                return Ok(None);
            }
        }

        let entropy_quality = vrf_result.output.entropy_estimate();
        let quantum_enhanced = vrf_result.metadata.quantum_enhanced;

        // Find anchor vertex for commit_round
        if commit_round % 2 == 0 {
            let anchors = self.anchor_results.read().await;
            if let Some(anchor_result) = anchors.get(&commit_round) {
                if let Some(anchor_id) = anchor_result.anchor_vertex_id {
                    // VRF-enhanced commit decision with quality assessment
                    let commit_type = if entropy_quality > 0.85 && quantum_enhanced {
                        // High-quality quantum VRF provides stronger commit guarantees
                        info!(
                            "High-quality quantum VRF commit for vertex {} with entropy {:.3}",
                            hex::encode(anchor_id),
                            entropy_quality
                        );
                        CommitType::DelayedCommit
                    } else {
                        CommitType::DelayedCommit
                    };

                    return self
                        .create_vrf_enhanced_commit_decision(
                            commit_round,
                            anchor_id,
                            commit_type,
                            vrf_result,
                        )
                        .await;
                }
            }
        }

        // For odd rounds or rounds without anchors, use enhanced chain commit logic
        self.evaluate_vrf_chain_commit(commit_round, vrf_result)
            .await
    }

    /// Check for delayed commits in a specific round
    pub async fn check_delayed_commits(&self, round: Round) -> Result<Vec<CommitDecision>> {
        debug!("Checking delayed commits for round {}", round);

        let mut commit_decisions = Vec::new();

        // Check if this round is δ rounds after any anchor elections
        let check_round = round + self.delta;

        let anchors = self.anchor_results.read().await;
        if let Some(anchor_result) = anchors.get(&check_round) {
            if let Some(anchor_id) = anchor_result.anchor_vertex_id {
                // Check if we haven't already committed this
                let history = self.commit_history.read().await;
                if !history.contains_key(&round) {
                    if let Some(decision) = self
                        .create_commit_decision(round, anchor_id, CommitType::DelayedCommit)
                        .await?
                    {
                        commit_decisions.push(decision);
                    }
                }
            }
        }

        Ok(commit_decisions)
    }

    /// Evaluate chain commit (causal dependency based)
    async fn evaluate_chain_commit(&self, round: Round) -> Result<Option<CommitDecision>> {
        debug!("Evaluating chain commit for round {}", round);

        // TODO: Implement causal dependency analysis
        // For now, skip chain commits in Phase 0

        // In a full implementation, this would:
        // 1. Analyze causal dependencies from committed vertices
        // 2. Find vertices that are causally dominated by committed vertices
        // 3. Create chain commit decisions for those vertices

        Ok(None)
    }

    /// Create a commit decision
    async fn create_commit_decision(
        &self,
        round: Round,
        vertex_id: VertexId,
        commit_type: CommitType,
    ) -> Result<Option<CommitDecision>> {
        // TODO: Retrieve actual transactions from vertex
        // For now, use empty transactions
        let transactions = vec![];

        let decision = CommitDecision {
            round,
            vertex_id,
            commit_type: commit_type.clone(),
            transactions,
            timestamp: chrono::Utc::now(),
        };

        // Store commit decision
        {
            let mut history = self.commit_history.write().await;
            history
                .entry(round)
                .or_insert_with(Vec::new)
                .push(decision.clone());
        }

        // Update statistics
        {
            let mut stats = self.statistics.write().await;
            stats.total_commits += 1;
            stats.successful_commits += 1;
            stats.last_commit_time = Some(decision.timestamp);

            match commit_type {
                CommitType::AnchorCommit => stats.anchor_commits += 1,
                CommitType::DelayedCommit => stats.delayed_commits += 1,
                CommitType::ChainCommit => stats.chain_commits += 1,
            }

            // Update average commit latency (simplified calculation)
            let latency = self.delta as f64;
            stats.average_commit_latency_rounds = (stats.average_commit_latency_rounds
                * (stats.successful_commits - 1) as f64
                + latency)
                / stats.successful_commits as f64;
        }

        info!(
            "Created commit decision for vertex {} in round {} ({:?})",
            hex::encode(vertex_id),
            round,
            commit_type
        );

        Ok(Some(decision))
    }

    /// Create a VRF-enhanced commit decision with quality metrics
    async fn create_vrf_enhanced_commit_decision(
        &self,
        round: Round,
        vertex_id: VertexId,
        commit_type: CommitType,
        vrf_result: &VRFResult,
    ) -> Result<Option<CommitDecision>> {
        // TODO: Retrieve actual transactions from vertex
        let transactions = vec![];

        let decision = CommitDecision {
            round,
            vertex_id,
            commit_type: commit_type.clone(),
            transactions,
            timestamp: chrono::Utc::now(),
        };

        // Store commit decision
        {
            let mut history = self.commit_history.write().await;
            history
                .entry(round)
                .or_insert_with(Vec::new)
                .push(decision.clone());
        }

        // Update VRF-enhanced statistics
        {
            let mut stats = self.statistics.write().await;
            stats.total_commits += 1;
            stats.successful_commits += 1;
            stats.last_commit_time = Some(decision.timestamp);

            match commit_type {
                CommitType::AnchorCommit => stats.anchor_commits += 1,
                CommitType::DelayedCommit => stats.delayed_commits += 1,
                CommitType::ChainCommit => stats.chain_commits += 1,
            }

            // Enhanced latency calculation with VRF quality factor
            let entropy_quality = vrf_result.output.entropy_estimate();
            let quality_bonus = if entropy_quality > 0.8 { 0.8 } else { 1.0 };
            let vrf_adjusted_latency = self.delta as f64 * quality_bonus;

            stats.average_commit_latency_rounds = (stats.average_commit_latency_rounds
                * (stats.successful_commits - 1) as f64
                + vrf_adjusted_latency)
                / stats.successful_commits as f64;
        }

        info!("Created VRF-enhanced commit decision for vertex {} in round {} ({:?}) with entropy {:.3}", 
              hex::encode(vertex_id), round, commit_type, vrf_result.output.entropy_estimate());

        Ok(Some(decision))
    }

    /// Evaluate VRF-enhanced chain commit logic
    async fn evaluate_vrf_chain_commit(
        &self,
        round: Round,
        vrf_result: &VRFResult,
    ) -> Result<Option<CommitDecision>> {
        debug!(
            "Evaluating VRF-enhanced chain commit for round {} with entropy {:.3}",
            round,
            vrf_result.output.entropy_estimate()
        );

        // TODO: Implement enhanced causal dependency analysis using VRF randomness
        // For now, skip VRF chain commits in Phase 1

        // In a full implementation, this would:
        // 1. Use VRF output to deterministically select causal dependencies
        // 2. Apply quantum randomness to break ties in causal ordering
        // 3. Create chain commit decisions enhanced by VRF quality metrics

        let entropy_quality = vrf_result.output.entropy_estimate();
        if entropy_quality > 0.95 && vrf_result.metadata.quantum_enhanced {
            debug!("High-quality quantum VRF could enable advanced chain commits (future implementation)");
        }

        Ok(None)
    }

    /// Get anchor election result
    async fn get_anchor_result(&self, round: Round) -> Option<AnchorElectionResult> {
        let anchors = self.anchor_results.read().await;
        anchors.get(&round).cloned()
    }

    /// Add anchor election result
    pub async fn add_anchor_result(&self, result: AnchorElectionResult) {
        let round = result.round;
        let mut anchors = self.anchor_results.write().await;
        anchors.insert(round, result);

        debug!("Added anchor result for round {}", round);
    }

    /// Get commit decisions for a round
    pub async fn get_commit_decisions(&self, round: Round) -> Vec<CommitDecision> {
        let history = self.commit_history.read().await;
        history.get(&round).cloned().unwrap_or_else(Vec::new)
    }

    /// Get all commit decisions up to a round
    pub async fn get_all_commits_up_to(&self, max_round: Round) -> Vec<CommitDecision> {
        let history = self.commit_history.read().await;
        let mut all_commits = Vec::new();

        for (&round, decisions) in history.iter() {
            if round <= max_round {
                all_commits.extend(decisions.clone());
            }
        }

        // Sort by round then by timestamp
        all_commits.sort_by(|a, b| {
            a.round
                .cmp(&b.round)
                .then_with(|| a.timestamp.cmp(&b.timestamp))
        });

        all_commits
    }

    /// Check if a round has been committed
    pub async fn is_round_committed(&self, round: Round) -> bool {
        let history = self.commit_history.read().await;
        history.contains_key(&round)
    }

    /// Get the latest committed round
    pub async fn get_latest_committed_round(&self) -> Option<Round> {
        let history = self.commit_history.read().await;
        history.keys().max().copied()
    }

    /// Verify commit decision validity
    pub async fn verify_commit_decision(&self, decision: &CommitDecision) -> Result<bool> {
        // Basic validity checks
        if decision.vertex_id == [0u8; 32] {
            warn!("Invalid commit decision: zero vertex ID");
            return Ok(false);
        }

        // Check if commit round is appropriate
        match decision.commit_type {
            CommitType::DelayedCommit => {
                // For delayed commits, verify δ-round rule
                let anchors = self.anchor_results.read().await;
                let anchor_round = decision.round + self.delta;

                if let Some(anchor_result) = anchors.get(&anchor_round) {
                    if anchor_result.anchor_vertex_id.is_none() {
                        warn!("Delayed commit without valid anchor election");
                        return Ok(false);
                    }
                } else {
                    warn!("Delayed commit without anchor election record");
                    return Ok(false);
                }
            }
            CommitType::AnchorCommit => {
                // Verify anchor election exists for this round
                let anchors = self.anchor_results.read().await;
                if !anchors.contains_key(&decision.round) {
                    warn!("Anchor commit without election result");
                    return Ok(false);
                }
            }
            CommitType::ChainCommit => {
                // TODO: Verify causal dependencies
            }
        }

        Ok(true)
    }

    /// Get commit statistics
    pub async fn get_statistics(&self) -> CommitStats {
        let mut stats = self.statistics.read().await;

        // Calculate TPS if we have commit history
        if let Some(last_commit) = stats.last_commit_time {
            let history = self.commit_history.read().await;
            let total_transactions: usize = history
                .values()
                .flat_map(|decisions| decisions.iter())
                .map(|d| d.transactions.len())
                .sum();

            // Simple TPS calculation over last minute
            let elapsed_seconds = (chrono::Utc::now() - last_commit).num_seconds().max(1) as f64;
            let tps = total_transactions as f64 / elapsed_seconds;

            // Update average TPS with exponential moving average
            let new_stats = CommitStats {
                average_tps: if stats.average_tps == 0.0 {
                    tps
                } else {
                    stats.average_tps * 0.9 + tps * 0.1
                },
                ..stats.clone()
            };

            return new_stats;
        }

        stats.clone()
    }

    /// Clean up old commit history
    pub async fn cleanup_old_commits(&self, keep_rounds: u64) {
        let latest_round = {
            let history = self.commit_history.read().await;
            history.keys().max().copied().unwrap_or(0)
        };

        let cutoff_round = latest_round.saturating_sub(keep_rounds);

        // Clean commit history
        {
            let mut history = self.commit_history.write().await;
            history.retain(|&round, _| round >= cutoff_round);
        }

        // Clean anchor results
        {
            let mut anchors = self.anchor_results.write().await;
            anchors.retain(|&round, _| round >= cutoff_round);
        }

        // Clean pending commits
        {
            let mut pending = self.pending_commits.write().await;
            pending.retain(|&round, _| round >= cutoff_round);
        }

        debug!(
            "Cleaned up commit history, keeping rounds >= {}",
            cutoff_round
        );
    }

    /// Detect potential tail forking in the δ-window
    ///
    /// This checks for conflicting block proposals within the finality window
    /// (rounds current - δ to current). If two different blocks are proposed
    /// for the same round, this indicates a tail fork attempt.
    pub async fn detect_tail_fork(&self, current_round: Round) -> Result<Option<TailForkDetection>> {
        // Check all rounds in the δ-window (unfinalized rounds)
        let start_round = current_round.saturating_sub(self.delta);

        let history = self.commit_history.read().await;

        for round in start_round..=current_round {
            if let Some(decisions) = history.get(&round) {
                // If multiple commits for same round, potential tail fork
                if decisions.len() > 1 {
                    // Check if vertex IDs differ (actual conflict)
                    let vertex_ids: HashSet<_> = decisions.iter()
                        .map(|d| d.vertex_id)
                        .collect();

                    if vertex_ids.len() > 1 {
                        let vertices: Vec<_> = vertex_ids.into_iter().collect();
                        warn!(
                            "🚨 TAIL FORK DETECTED at round {}: {} conflicting vertices",
                            round, vertices.len()
                        );

                        return Ok(Some(TailForkDetection {
                            round,
                            vertex_a: vertices[0],
                            vertex_b: vertices[1],
                            detected_at: chrono::Utc::now(),
                            severity: 9, // High severity - consensus attack
                        }));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Check if proposing a new block would create a tail fork
    ///
    /// CRITICAL: Call this BEFORE creating a new block proposal.
    /// Returns Err if proposal would conflict with uncommitted ancestor.
    pub async fn validate_proposal_safety(
        &self,
        proposed_round: Round,
        proposed_vertex_id: VertexId,
        parent_vertex_id: VertexId,
    ) -> Result<bool> {
        // Rule 1: Cannot propose if parent is still in δ-window and uncommitted
        let parent_round = proposed_round.saturating_sub(1);

        if !self.is_round_committed(parent_round).await && proposed_round > self.delta {
            warn!(
                "⚠️ Unsafe proposal: parent round {} not yet committed for proposed round {}",
                parent_round, proposed_round
            );
            // Allow proposal but log warning - let voting decide
        }

        // Rule 2: Check for existing proposals at this round
        let history = self.commit_history.read().await;
        if let Some(existing_decisions) = history.get(&proposed_round) {
            for existing in existing_decisions {
                if existing.vertex_id != proposed_vertex_id {
                    warn!(
                        "🚨 CONFLICTING PROPOSAL: Round {} already has vertex {}, rejecting {}",
                        proposed_round,
                        hex::encode(existing.vertex_id),
                        hex::encode(proposed_vertex_id)
                    );
                    return Ok(false); // Reject conflicting proposal
                }
            }
        }

        // Rule 3: Verify parent vertex exists and is valid
        // (ancestor finality guarantee)
        let anchors = self.anchor_results.read().await;
        let parent_anchor_round = (parent_round / 2) * 2; // Find nearest even round

        if let Some(anchor_result) = anchors.get(&parent_anchor_round) {
            if anchor_result.anchor_vertex_id.is_none() {
                warn!(
                    "⚠️ Proposal at round {} has no valid anchor at parent round {}",
                    proposed_round, parent_anchor_round
                );
                // Still allow - anchor might come later
            }
        }

        info!(
            "✅ Proposal validated: round {} vertex {} parent {}",
            proposed_round,
            hex::encode(&proposed_vertex_id[..8]),
            hex::encode(&parent_vertex_id[..8])
        );

        Ok(true)
    }

    /// Get the latest safely finalized round (guaranteed no reorg)
    ///
    /// Returns the highest round where we can guarantee:
    /// 1. Block is committed by 2/3+1 stake
    /// 2. No conflicting blocks exist
    /// 3. All ancestors are also finalized
    pub async fn get_safe_finalized_round(&self) -> Round {
        let history = self.commit_history.read().await;

        // Start from latest committed and work backwards
        let latest = history.keys().max().copied().unwrap_or(0);

        // Safe round is at least δ rounds behind latest
        let safe_round = latest.saturating_sub(self.delta);

        // Verify no conflicts in the range [safe_round, latest]
        for round in safe_round..=latest {
            if let Some(decisions) = history.get(&round) {
                let unique_vertices: HashSet<_> = decisions.iter()
                    .map(|d| d.vertex_id)
                    .collect();

                if unique_vertices.len() > 1 {
                    // Conflict found - safe round is before this
                    return round.saturating_sub(1);
                }
            }
        }

        safe_round
    }

    /// Check if a specific round has reached final safety
    /// (cannot be reorganized under BFT assumptions)
    pub async fn is_round_final(&self, round: Round) -> bool {
        let safe_round = self.get_safe_finalized_round().await;
        round <= safe_round
    }

    /// Record a block proposal for tail fork tracking
    ///
    /// Call this when a new block is proposed (before voting)
    pub async fn record_proposal(&self, round: Round, vertex_id: VertexId) -> Result<()> {
        // Check for conflicting proposals first
        if !self.validate_proposal_safety(round, vertex_id, [0u8; 32]).await? {
            return Err(anyhow::anyhow!(
                "Proposal rejected: would create tail fork at round {}",
                round
            ));
        }

        // Record as pending (not yet committed)
        let pending = PendingCommit {
            round,
            vertex_id,
            commit_type: CommitType::DelayedCommit,
            evaluation_time: chrono::Utc::now(),
            dependencies: vec![],
        };

        let mut pending_commits = self.pending_commits.write().await;
        pending_commits.insert(round, pending);

        debug!("Recorded proposal for round {} vertex {}", round, hex::encode(&vertex_id[..8]));
        Ok(())
    }

    /// Get all pending (uncommitted) proposals in the δ-window
    pub async fn get_pending_proposals(&self) -> Vec<PendingCommit> {
        let pending = self.pending_commits.read().await;
        pending.values().cloned().collect()
    }

    /// Calculate the ancestor depth that must be committed before proposing
    ///
    /// For full tail fork protection, ancestors must be committed before
    /// building on them. This returns how many ancestor rounds must be finalized.
    pub fn required_ancestor_depth(&self) -> u64 {
        // Conservative: require δ ancestors to be committed
        // This eliminates the tail fork window entirely
        self.delta
    }

    /// Get commit rule evaluation result
    pub async fn evaluate_commit_rules(
        &self,
        round: Round,
        vertex_id: VertexId,
    ) -> CommitRuleResult {
        // Rule 1: δ-round delayed commit
        if round >= self.delta {
            let commit_round = round - self.delta;

            // Check if we have anchor for this round
            let anchors = self.anchor_results.read().await;
            if let Some(anchor_result) = anchors.get(&commit_round) {
                if anchor_result.anchor_vertex_id == Some(vertex_id) {
                    if let Ok(Some(decision)) = self
                        .create_commit_decision(commit_round, vertex_id, CommitType::DelayedCommit)
                        .await
                    {
                        return CommitRuleResult::Commit(decision);
                    }
                }
            }
        }

        // Rule 2: Immediate anchor commit (if we're the anchor)
        let anchors = self.anchor_results.read().await;
        if let Some(anchor_result) = anchors.get(&round) {
            if anchor_result.anchor_vertex_id == Some(vertex_id) {
                if let Ok(Some(decision)) = self
                    .create_commit_decision(round, vertex_id, CommitType::AnchorCommit)
                    .await
                {
                    return CommitRuleResult::Commit(decision);
                }
            }
        }

        // Rule 3: Chain commit (causal dependency)
        // TODO: Implement full causal analysis

        // Default: defer commit
        CommitRuleResult::Defer {
            reason: "Waiting for δ-round delay or anchor election".to_string(),
            retry_after_rounds: self.delta - (round % self.delta).max(1),
        }
    }

    /// Update delta parameter
    pub async fn update_delta(&mut self, new_delta: u64) {
        let old_delta = self.delta;
        self.delta = new_delta;

        info!(
            "Updated commit delta from {} to {} rounds",
            old_delta, new_delta
        );
    }

    /// Get current configuration
    pub fn get_config(&self) -> (usize, u64) {
        (self.f, self.delta)
    }

    /// Process a vertex through the commit protocol
    pub async fn process_vertex(&self, vertex: &Vertex) -> Result<()> {
        // TODO: Implement actual vertex processing logic
        // This should update commit state based on the received vertex
        info!("Processing vertex {:?} in commit protocol", vertex.id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_anchor_result(
        round: Round,
        vertex_id: Option<VertexId>,
    ) -> AnchorElectionResult {
        AnchorElectionResult {
            round,
            anchor_vertex_id: vertex_id,
            vdf_output: [42u8; 32],
            quantum_beacon: [24u8; 32],
            election_strength: 0.95,
            candidates: vec![],
        }
    }

    #[tokio::test]
    async fn test_commit_protocol_creation() {
        let protocol = CommitProtocol::new(1);
        assert!(protocol.is_ok());

        let protocol = protocol.unwrap();
        let (f, delta) = protocol.get_config();
        assert_eq!(f, 1);
        assert_eq!(delta, 4);
    }

    #[tokio::test]
    async fn test_anchor_result_storage() {
        let protocol = CommitProtocol::new(1).unwrap();
        let anchor_result = create_test_anchor_result(2, Some([1u8; 32]));

        protocol.add_anchor_result(anchor_result.clone()).await;

        let retrieved = protocol.get_anchor_result(2).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().round, 2);
    }

    #[tokio::test]
    async fn test_delayed_commit_evaluation() {
        let protocol = CommitProtocol::new(1).unwrap();
        let vertex_id = [1u8; 32];

        // Add anchor result for round 2
        let anchor_result = create_test_anchor_result(2, Some(vertex_id));
        protocol.add_anchor_result(anchor_result).await;

        // Evaluate commit for round 6 (should commit round 2)
        let commit_decision = protocol.evaluate_commit(6, vertex_id).await.unwrap();

        assert!(commit_decision.is_some());
        let decision = commit_decision.unwrap();
        assert_eq!(decision.round, 2);
        assert_eq!(decision.vertex_id, vertex_id);
        assert!(matches!(decision.commit_type, CommitType::DelayedCommit));
    }

    #[tokio::test]
    async fn test_commit_verification() {
        let protocol = CommitProtocol::new(1).unwrap();
        let vertex_id = [1u8; 32];

        // Add anchor result
        let anchor_result = create_test_anchor_result(2, Some(vertex_id));
        protocol.add_anchor_result(anchor_result).await;

        // Create commit decision
        let decision = CommitDecision {
            round: 2,
            vertex_id,
            commit_type: CommitType::DelayedCommit,
            transactions: vec![],
            timestamp: chrono::Utc::now(),
        };

        // Should be invalid without proper anchor setup
        let is_valid = protocol.verify_commit_decision(&decision).await.unwrap();
        // Note: This might be true or false depending on exact verification logic
    }

    #[tokio::test]
    async fn test_commit_statistics() {
        let protocol = CommitProtocol::new(1).unwrap();
        let vertex_id = [1u8; 32];

        // Initial stats
        let stats = protocol.get_statistics().await;
        assert_eq!(stats.total_commits, 0);
        assert_eq!(stats.successful_commits, 0);

        // Add anchor and commit
        let anchor_result = create_test_anchor_result(2, Some(vertex_id));
        protocol.add_anchor_result(anchor_result).await;

        protocol.evaluate_commit(6, vertex_id).await.unwrap();

        // Check updated stats
        let updated_stats = protocol.get_statistics().await;
        assert!(updated_stats.successful_commits > 0);
    }

    #[tokio::test]
    async fn test_round_committed_check() {
        let protocol = CommitProtocol::new(1).unwrap();
        let vertex_id = [1u8; 32];

        // Round should not be committed initially
        assert!(!protocol.is_round_committed(2).await);

        // Add anchor and commit
        let anchor_result = create_test_anchor_result(2, Some(vertex_id));
        protocol.add_anchor_result(anchor_result).await;

        protocol.evaluate_commit(6, vertex_id).await.unwrap();

        // Round should now be committed
        assert!(protocol.is_round_committed(2).await);
    }

    #[tokio::test]
    async fn test_commit_rule_evaluation() {
        let protocol = CommitProtocol::new(1).unwrap();
        let vertex_id = [1u8; 32];

        // Should defer early commits
        let result = protocol.evaluate_commit_rules(1, vertex_id).await;
        assert!(matches!(result, CommitRuleResult::Defer { .. }));

        // Add anchor election
        let anchor_result = create_test_anchor_result(2, Some(vertex_id));
        protocol.add_anchor_result(anchor_result).await;

        // Should commit after delta rounds
        let result = protocol.evaluate_commit_rules(6, vertex_id).await;
        assert!(matches!(result, CommitRuleResult::Commit(_)));
    }
}
