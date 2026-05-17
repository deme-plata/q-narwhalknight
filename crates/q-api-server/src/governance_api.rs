//! Governance API endpoints for Proof-of-Contribution voting system
//!
//! This module provides REST API endpoints for:
//! - Creating governance proposals (parameter changes, treasury, upgrades)
//! - Submitting weighted votes with mining contributions
//! - Calculating proposal results with voting power
//! - Querying reputation and contribution statistics

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use q_governance::{
    ContributionStats, GovernanceCoordinator, MiningContribution, Proposal, ProposalResult,
    ProposalType, WeightedVote,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info};

use crate::AppState;

/// Request to create a new governance proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateProposalRequest {
    pub title: String,
    pub description: String,
    pub proposer: String, // Hex-encoded address
    pub options: Vec<String>,
    pub voting_duration_seconds: u64,
    pub proposal_type: String, // "parameter_change", "treasury_allocation", "protocol_upgrade"
    pub required_quorum: u128,
}

/// Request to submit a vote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitVoteRequest {
    pub proposal_id: String,
    pub voter_address: String, // Hex-encoded
    pub vote_choice: String,
    pub token_stake: u128,
    pub mining_contribution: Option<MiningContributionDto>,
    pub signature: String, // Hex-encoded
}

/// Mining contribution data transfer object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningContributionDto {
    pub total_hashes: u128,
    pub contribution_period_start: u64,
    pub contribution_period_end: u64,
    // Merkle proofs omitted for now - will be validated on-chain
}

/// Query parameters for listing proposals
#[derive(Debug, Deserialize)]
pub struct ListProposalsQuery {
    pub status: Option<String>, // "active", "completed", "all"
    pub proposal_type: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Response for proposal creation
#[derive(Debug, Serialize)]
pub struct CreateProposalResponse {
    pub proposal_id: String,
    pub message: String,
}

/// Response for vote submission
#[derive(Debug, Serialize)]
pub struct SubmitVoteResponse {
    pub message: String,
    pub voting_power: u128,
    pub bonus_percent: f64,
}

/// Response for proposal results
#[derive(Debug, Serialize)]
pub struct ProposalResultsResponse {
    pub proposal_id: String,
    pub proposal: Proposal,
    pub results: ProposalResult,
    pub is_quorum_met: bool,
    pub status: String, // "active", "passed", "rejected"
}

/// Response for contribution stats
#[derive(Debug, Serialize)]
pub struct ContributionStatsResponse {
    pub address: String,
    pub stats: ContributionStats,
    pub reputation_multiplier: f64,
}

/// Create governance router with all endpoints
pub fn create_governance_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/proposals", post(create_proposal))
        .route("/proposals", get(list_proposals))
        .route("/proposals/:id", get(get_proposal))
        .route("/votes", post(submit_vote))
        .route("/results/:id", get(get_results))
        .route("/stats/:address", get(get_contribution_stats))
        .route("/reputation/:address", get(get_reputation))
}

/// POST /api/governance/proposals
/// Create a new governance proposal
async fn create_proposal(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateProposalRequest>,
) -> Result<Json<CreateProposalResponse>, (StatusCode, String)> {
    info!("📜 Creating governance proposal: {}", req.title);

    // Parse proposer address
    let proposer_bytes = hex::decode(&req.proposer).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Invalid proposer address: {}", e),
        )
    })?;

    if proposer_bytes.len() != 32 {
        return Err((
            StatusCode::BAD_REQUEST,
            "Proposer address must be 32 bytes".to_string(),
        ));
    }

    let mut proposer = [0u8; 32];
    proposer.copy_from_slice(&proposer_bytes);

    // Parse proposal type
    let proposal_type = match req.proposal_type.to_lowercase().as_str() {
        "parameter_change" => ProposalType::ParameterChange,
        "treasury_allocation" => ProposalType::TreasuryAllocation,
        "protocol_upgrade" => ProposalType::ProtocolUpgrade,
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("Invalid proposal type: {}", req.proposal_type),
            ))
        }
    };

    // Create proposal
    let now = chrono::Utc::now().timestamp() as u64;
    let proposal_id = format!(
        "prop-{}-{}",
        now,
        hex::encode(&proposer[..8]) // First 8 bytes of proposer address
    );

    let proposal = Proposal {
        id: proposal_id.clone(),
        title: req.title,
        description: req.description,
        proposer,
        options: req.options,
        voting_start: now,
        voting_end: now + req.voting_duration_seconds,
        proposal_type,
        required_quorum: req.required_quorum,
    };

    // Get governance coordinator from state
    let governance = get_governance_coordinator(&state)?;

    // Create proposal
    governance
        .create_proposal(proposal)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    info!("✅ Created proposal: {}", proposal_id);

    Ok(Json(CreateProposalResponse {
        proposal_id,
        message: "Proposal created successfully".to_string(),
    }))
}

/// GET /api/governance/proposals
/// List all proposals with optional filtering
async fn list_proposals(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListProposalsQuery>,
) -> Result<Json<Vec<Proposal>>, (StatusCode, String)> {
    debug!("📋 Listing governance proposals");

    let governance = get_governance_coordinator(&state)?;
    let mut proposals = governance.get_all_proposals().await;

    // Filter by status
    if let Some(status) = query.status {
        let now = chrono::Utc::now().timestamp() as u64;
        proposals.retain(|p| match status.as_str() {
            "active" => p.voting_end > now,
            "completed" => p.voting_end <= now,
            "all" => true,
            _ => true,
        });
    }

    // Filter by type
    if let Some(ptype) = query.proposal_type {
        proposals
            .retain(|p| format!("{:?}", p.proposal_type).to_lowercase() == ptype.to_lowercase());
    }

    // Apply pagination
    let offset = query.offset.unwrap_or(0);
    let limit = query.limit.unwrap_or(50).min(100); // Max 100 per page

    let paginated: Vec<_> = proposals.into_iter().skip(offset).take(limit).collect();

    Ok(Json(paginated))
}

/// GET /api/governance/proposals/:id
/// Get specific proposal details
async fn get_proposal(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Proposal>, (StatusCode, String)> {
    debug!("🔍 Getting proposal: {}", id);

    let governance = get_governance_coordinator(&state)?;
    let proposal = governance
        .get_proposal(&id)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    Ok(Json(proposal))
}

/// POST /api/governance/votes
/// Submit a weighted vote with optional mining contribution
async fn submit_vote(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SubmitVoteRequest>,
) -> Result<Json<SubmitVoteResponse>, (StatusCode, String)> {
    info!("🗳️  Submitting vote for proposal: {}", req.proposal_id);

    // Parse voter address
    let voter_bytes = hex::decode(&req.voter_address).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Invalid voter address: {}", e),
        )
    })?;

    if voter_bytes.len() != 32 {
        return Err((
            StatusCode::BAD_REQUEST,
            "Voter address must be 32 bytes".to_string(),
        ));
    }

    let mut voter_address = [0u8; 32];
    voter_address.copy_from_slice(&voter_bytes);

    // Parse signature
    let signature = hex::decode(&req.signature)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid signature: {}", e)))?;

    // Convert mining contribution DTO if provided
    let mining_contribution = req.mining_contribution.map(|dto| MiningContribution {
        solutions: vec![], // Solutions will be validated on-chain
        total_hashes: dto.total_hashes,
        merkle_proofs: vec![], // Proofs will be validated on-chain
        contribution_period: (dto.contribution_period_start, dto.contribution_period_end),
    });

    // Create vote
    let vote = WeightedVote {
        proposal_id: req.proposal_id.clone(),
        voter_address,
        vote_choice: req.vote_choice,
        token_stake: req.token_stake,
        mining_contribution: mining_contribution.clone(),
        timestamp: chrono::Utc::now().timestamp() as u64,
        signature,
    };

    // Get governance coordinator
    let governance = get_governance_coordinator(&state)?;

    // Submit vote
    governance
        .submit_vote(vote.clone())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Calculate voting power for response
    let voting_power = governance.calculate_voting_power(&vote);
    let bonus_percent = if let Some(contrib) = &mining_contribution {
        let power_without = req.token_stake;
        ((voting_power as f64 / power_without as f64) - 1.0) * 100.0
    } else {
        0.0
    };

    info!(
        "✅ Vote submitted: power={}, bonus={:.2}%",
        voting_power, bonus_percent
    );

    Ok(Json(SubmitVoteResponse {
        message: "Vote submitted successfully".to_string(),
        voting_power,
        bonus_percent,
    }))
}

/// GET /api/governance/results/:id
/// Calculate and return proposal results
async fn get_results(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ProposalResultsResponse>, (StatusCode, String)> {
    info!("📊 Calculating results for proposal: {}", id);

    let governance = get_governance_coordinator(&state)?;

    // Get proposal
    let proposal = governance
        .get_proposal(&id)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    // Calculate results
    let results = governance
        .calculate_results(&id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Determine status
    let now = chrono::Utc::now().timestamp() as u64;
    let is_active = proposal.voting_end > now;
    let is_quorum_met = results.total_voting_power >= proposal.required_quorum;

    let status = if is_active {
        "active".to_string()
    } else if is_quorum_met && results.winning_option.is_some() {
        "passed".to_string()
    } else {
        "rejected".to_string()
    };

    Ok(Json(ProposalResultsResponse {
        proposal_id: id,
        proposal,
        results,
        is_quorum_met,
        status,
    }))
}

/// GET /api/governance/stats/:address
/// Get contribution statistics for an address
async fn get_contribution_stats(
    State(state): State<Arc<AppState>>,
    Path(address): Path<String>,
) -> Result<Json<ContributionStatsResponse>, (StatusCode, String)> {
    debug!("📈 Getting contribution stats for: {}", address);

    let addr_bytes = hex::decode(&address)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid address: {}", e)))?;

    if addr_bytes.len() != 32 {
        return Err((
            StatusCode::BAD_REQUEST,
            "Address must be 32 bytes".to_string(),
        ));
    }

    let mut addr = [0u8; 32];
    addr.copy_from_slice(&addr_bytes);

    let governance = get_governance_coordinator(&state)?;

    // Get contribution stats
    let stats = governance.get_contribution_stats(&addr).await;

    // Get reputation multiplier
    let reputation_multiplier = governance.get_reputation_multiplier(&addr).await;

    Ok(Json(ContributionStatsResponse {
        address,
        stats,
        reputation_multiplier,
    }))
}

/// GET /api/governance/reputation/:address
/// Get miner reputation details
async fn get_reputation(
    State(state): State<Arc<AppState>>,
    Path(address): Path<String>,
) -> Result<Json<q_governance::MinerReputation>, (StatusCode, String)> {
    debug!("🏆 Getting reputation for: {}", address);

    let addr_bytes = hex::decode(&address)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid address: {}", e)))?;

    if addr_bytes.len() != 32 {
        return Err((
            StatusCode::BAD_REQUEST,
            "Address must be 32 bytes".to_string(),
        ));
    }

    let mut addr = [0u8; 32];
    addr.copy_from_slice(&addr_bytes);

    let governance = get_governance_coordinator(&state)?;

    let reputation = governance.get_reputation(&addr).await.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            "No reputation data found".to_string(),
        )
    })?;

    Ok(Json(reputation))
}

/// Helper function to get governance coordinator from AppState
/// v2.4.0-beta: Now uses persistent coordinator from AppState (no longer creates new instance)
fn get_governance_coordinator(
    state: &Arc<AppState>,
) -> Result<Arc<GovernanceCoordinator>, (StatusCode, String)> {
    Ok(state.governance_coordinator.clone())
}
