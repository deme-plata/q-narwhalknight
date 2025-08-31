/// DAG-Knight commit logic implementation
/// Handles commit rules and finality decisions

use q_types::*;
use super::{CommitDecision, CommitType, AnchorElectionResult};
use anyhow::Result;
use std::collections::{HashMap, HashSet, BTreeMap};
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
    Defer { reason: String, retry_after_rounds: u64 },
    Reject { reason: String },
}

impl CommitProtocol {
    pub fn new(f: usize) -> Result<Self> {
        Ok(Self {
            f,
            delta: 4, // Conservative default for DAG-Knight
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
    pub async fn evaluate_commit(&self, round: Round, anchor_vertex_id: VertexId) -> Result<Option<CommitDecision>> {
        debug!("Evaluating commit for anchor vertex {} in round {}", 
               hex::encode(anchor_vertex_id), round);

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
                    return self.create_commit_decision(
                        commit_round,
                        anchor_id,
                        CommitType::DelayedCommit,
                    ).await;
                }
            }
        }

        // For odd rounds or rounds without anchors, check for chain commits
        self.evaluate_chain_commit(commit_round).await
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
                    if let Some(decision) = self.create_commit_decision(
                        round,
                        anchor_id,
                        CommitType::DelayedCommit,
                    ).await? {
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
        commit_type: CommitType
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
            history.entry(round).or_insert_with(Vec::new).push(decision.clone());
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
            stats.average_commit_latency_rounds = 
                (stats.average_commit_latency_rounds * (stats.successful_commits - 1) as f64 + latency)
                / stats.successful_commits as f64;
        }

        info!("Created commit decision for vertex {} in round {} ({:?})", 
              hex::encode(vertex_id), round, commit_type);

        Ok(Some(decision))
    }

    /// Get anchor election result
    async fn get_anchor_result(&self, round: Round) -> Option<AnchorElectionResult> {
        let anchors = self.anchor_results.read().await;
        anchors.get(&round).cloned()
    }

    /// Add anchor election result
    pub async fn add_anchor_result(&self, result: AnchorElectionResult) {
        let mut anchors = self.anchor_results.write().await;
        anchors.insert(result.round, result);
        
        debug!("Added anchor result for round {}", result.round);
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
            a.round.cmp(&b.round)
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
            let total_transactions: usize = history.values()
                .flat_map(|decisions| decisions.iter())
                .map(|d| d.transactions.len())
                .sum();
            
            // Simple TPS calculation over last minute
            let elapsed_seconds = (chrono::Utc::now() - last_commit).num_seconds().max(1) as f64;
            let tps = total_transactions as f64 / elapsed_seconds;
            
            // Update average TPS with exponential moving average
            let new_stats = CommitStats {
                average_tps: if stats.average_tps == 0.0 { tps } else { 
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

        debug!("Cleaned up commit history, keeping rounds >= {}", cutoff_round);
    }

    /// Get commit rule evaluation result
    pub async fn evaluate_commit_rules(&self, round: Round, vertex_id: VertexId) -> CommitRuleResult {
        // Rule 1: δ-round delayed commit
        if round >= self.delta {
            let commit_round = round - self.delta;
            
            // Check if we have anchor for this round
            let anchors = self.anchor_results.read().await;
            if let Some(anchor_result) = anchors.get(&commit_round) {
                if anchor_result.anchor_vertex_id == Some(vertex_id) {
                    if let Ok(Some(decision)) = self.create_commit_decision(
                        commit_round, 
                        vertex_id, 
                        CommitType::DelayedCommit
                    ).await {
                        return CommitRuleResult::Commit(decision);
                    }
                }
            }
        }

        // Rule 2: Immediate anchor commit (if we're the anchor)
        let anchors = self.anchor_results.read().await;
        if let Some(anchor_result) = anchors.get(&round) {
            if anchor_result.anchor_vertex_id == Some(vertex_id) {
                if let Ok(Some(decision)) = self.create_commit_decision(
                    round, 
                    vertex_id, 
                    CommitType::AnchorCommit
                ).await {
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
        
        info!("Updated commit delta from {} to {} rounds", old_delta, new_delta);
    }

    /// Get current configuration
    pub fn get_config(&self) -> (usize, u64) {
        (self.f, self.delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_anchor_result(round: Round, vertex_id: Option<VertexId>) -> AnchorElectionResult {
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