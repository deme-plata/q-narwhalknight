/// Core data types for the Testnet Bounty Protocol
///
/// This module defines all data structures for tracking testnet participation,
/// scoring activities, and managing mainnet reward claims.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Testnet user with identity binding to mainnet wallet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestnetUser {
    /// Unique user identifier
    pub user_id: Uuid,

    /// Testnet wallet address (32-byte QNK address)
    pub testnet_address: [u8; 32],

    /// Optional mainnet wallet address for reward claims
    pub mainnet_address: Option<[u8; 32]>,

    /// Registration timestamp
    pub registration_date: i64,

    /// Social account bindings for verification
    pub social_accounts: SocialAccounts,

    /// User tier based on contribution level
    pub tier: BountyTier,

    /// Total accumulated score
    pub total_score: f64,

    /// Category-specific score breakdown
    pub category_scores: CategoryScores,

    /// Early participation multiplier
    pub early_multiplier: f64,

    /// Consistency bonus multiplier
    pub consistency_bonus: f64,

    /// Current leaderboard rank
    pub rank: Option<u32>,

    /// KYC verification status (for large rewards)
    pub kyc_verified: bool,
}

/// Social account bindings for identity verification
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SocialAccounts {
    /// code.quillon.xyz username and OAuth token
    #[serde(alias = "github")]
    pub code_quillon: Option<SocialBinding>,

    /// Twitter/X handle and OAuth token
    pub twitter: Option<SocialBinding>,

    /// Discord user ID and verification
    pub discord: Option<SocialBinding>,
}

/// Individual social media account binding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialBinding {
    /// Platform username or user ID
    pub username: String,

    /// OAuth access token (encrypted at rest)
    pub access_token: Option<String>,

    /// Verification timestamp
    pub verified_at: i64,

    /// Verification status
    pub verified: bool,
}

/// Bounty tier system based on contribution level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BountyTier {
    /// Top 1% - 20% of reward pool, 6-month vesting
    Pioneer,

    /// Top 10% - 30% of reward pool, 3-month vesting
    Contributor,

    /// Top 50% - 40% of reward pool, 1-month vesting
    Participant,

    /// All valid users - 10% of reward pool, no vesting
    Supporter,
}

impl BountyTier {
    /// Get the reward pool percentage for this tier
    pub fn reward_percentage(&self) -> f64 {
        match self {
            BountyTier::Pioneer => 0.20,
            BountyTier::Contributor => 0.30,
            BountyTier::Participant => 0.40,
            BountyTier::Supporter => 0.10,
        }
    }

    /// Get the vesting period in months
    pub fn vesting_months(&self) -> u32 {
        match self {
            BountyTier::Pioneer => 6,
            BountyTier::Contributor => 3,
            BountyTier::Participant => 1,
            BountyTier::Supporter => 0,
        }
    }
}

/// Category-specific score breakdown
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CategoryScores {
    /// Node operations: uptime, blocks, governance (35% weight) // v8.6.0: raised from 30% to prioritize infrastructure
    pub node_ops: f64,

    /// Transaction volume and diversity (10% weight) // v8.6.0: reduced from 25% to discourage tx farming
    pub transactions: f64,

    /// Bug reports and fixes (30% weight) // v8.6.0: raised from 20% to prioritize security
    pub bug_reports: f64,

    /// Community contributions: docs, tutorials (15% weight)
    pub community: f64,

    /// Social media engagement and education (10% weight)
    pub social: f64,
}

impl CategoryScores {
    /// Calculate the weighted total score
    /// v8.6.0: rebalanced weights — infrastructure 35%, security 30%, tx 10%
    pub fn calculate_total(&self) -> f64 {
        0.35 * self.node_ops
            + 0.10 * self.transactions
            + 0.30 * self.bug_reports
            + 0.15 * self.community
            + 0.10 * self.social
    }
}

/// Activity category for tracking testnet participation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ActivityCategory {
    NodeOps,
    Transactions,
    BugReports,
    Community,
    Social,
}

impl ActivityCategory {
    /// Get the scoring weight for this category
    /// v8.6.0: rebalanced — infra 35%, security 30%, tx 10%
    pub fn weight(&self) -> f64 {
        match self {
            ActivityCategory::NodeOps => 0.35,       // v8.6.0: was 0.30
            ActivityCategory::Transactions => 0.10,  // v8.6.0: was 0.25
            ActivityCategory::BugReports => 0.30,    // v8.6.0: was 0.20
            ActivityCategory::Community => 0.15,
            ActivityCategory::Social => 0.10,
        }
    }

    /// Get the maximum achievable score for this category
    pub fn max_score(&self) -> f64 {
        match self {
            ActivityCategory::NodeOps => 300.0,
            ActivityCategory::Transactions => 250.0,
            ActivityCategory::BugReports => 200.0,
            ActivityCategory::Community => 150.0,
            ActivityCategory::Social => 100.0,
        }
    }
}

/// Node operation metrics for scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetrics {
    pub user_id: Uuid,
    pub uptime_percentage: f64,
    pub blocks_produced: u64,
    pub governance_votes: u64,
    pub peer_count: u32,
    pub timestamp: i64,
}

/// Transaction activity for scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionActivity {
    pub user_id: Uuid,
    pub tx_hash: String,
    pub value: u64,
    pub contract_interactions: u32,
    pub dex_swaps: u32,
    pub timestamp: i64,
}

/// Bug report with severity weighting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BugReport {
    pub user_id: Uuid,
    /// URL to issue on code.quillon.xyz or any tracker
    #[serde(alias = "github_issue_url")]
    pub issue_url: String,
    pub severity: BugSeverity,
    pub status: BugStatus,
    pub bounty_awarded: u64,
    pub description: String,
    pub timestamp: i64,
}

/// Bug severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BugSeverity {
    Critical,   // 10% of bounty pool
    High,       // 5% of bounty pool
    Medium,     // 2% of bounty pool
    Low,        // 1% of bounty pool
}

impl BugSeverity {
    /// Get the score multiplier for this severity
    /// v8.6.0: doubled Critical (200) and High (100) to incentivize security research
    pub fn score_multiplier(&self) -> f64 {
        match self {
            BugSeverity::Critical => 200.0, // v8.6.0: was 100.0
            BugSeverity::High => 100.0,     // v8.6.0: was 50.0
            BugSeverity::Medium => 20.0,
            BugSeverity::Low => 10.0,
        }
    }
}

/// Bug report status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BugStatus {
    Submitted,
    UnderReview,
    Verified,
    Duplicate,
    Invalid,
    Fixed,
}

/// Community contribution tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityContribution {
    pub user_id: Uuid,
    pub contribution_type: ContributionType,
    pub url: String,
    pub impact_score: f64,
    pub verified: bool,
    pub timestamp: i64,
}

/// Types of community contributions
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ContributionType {
    Documentation,
    Tutorial,
    CodeExample,
    CommunitySupport,
    Translation,
    ToolDevelopment,
}

impl ContributionType {
    /// Get the base score for this contribution type
    pub fn base_score(&self) -> f64 {
        match self {
            ContributionType::Documentation => 50.0,
            ContributionType::Tutorial => 40.0,
            ContributionType::CodeExample => 30.0,
            ContributionType::CommunitySupport => 20.0,
            ContributionType::Translation => 25.0,
            ContributionType::ToolDevelopment => 60.0,
        }
    }
}

/// Social media activity tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialActivity {
    pub user_id: Uuid,
    pub platform: SocialPlatform,
    pub activity_type: SocialActivityType,
    pub content_url: String,
    pub engagement_score: f64,
    pub verified: bool,
    pub timestamp: i64,
}

/// Supported social media platforms
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SocialPlatform {
    Twitter,
    #[serde(alias = "GitHub")]
    CodeQuillon,
    Discord,
    Medium,
    YouTube,
}

/// Types of social media activities
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SocialActivityType {
    Tweet,
    Thread,
    Article,
    Video,
    DiscordMessage,
    #[serde(alias = "GitHubPR")]
    MergeRequest,
    #[serde(alias = "GitHubIssue")]
    CodeIssue,
}

impl SocialActivityType {
    /// Get the base score for this activity type
    pub fn base_score(&self) -> f64 {
        match self {
            SocialActivityType::Video => 50.0,
            SocialActivityType::Article => 40.0,
            SocialActivityType::Thread => 30.0,
            SocialActivityType::Tweet => 10.0,
            SocialActivityType::DiscordMessage => 5.0,
            SocialActivityType::MergeRequest => 35.0,
            SocialActivityType::CodeIssue => 15.0,
        }
    }
}

/// Claim status for mainnet distribution
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ClaimStatus {
    PendingVerification,
    ReadyToClaim,
    Claimed { tx_hash: [u8; 32], timestamp: i64 },
    Disputed { dispute_id: Uuid },
}

/// Final user score with Merkle proof for mainnet claims
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalScore {
    pub user_id: Uuid,
    pub testnet_address: [u8; 32],
    pub mainnet_address: Option<[u8; 32]>,
    pub total_score: f64,
    pub category_breakdown: CategoryScores,
    pub rank: u32,
    pub tier: BountyTier,
    pub reward_amount: u64,
    pub merkle_proof: Option<Vec<[u8; 32]>>,
    pub claim_status: ClaimStatus,
}

/// Dispute for score challenges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreDispute {
    pub dispute_id: Uuid,
    pub user_id: Uuid,
    pub disputed_category: ActivityCategory,
    pub reason: String,
    pub evidence_urls: Vec<String>,
    pub status: DisputeStatus,
    pub admin_response: Option<String>,
    pub submitted_at: i64,
    pub resolved_at: Option<i64>,
}

/// Dispute status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DisputeStatus {
    Submitted,
    UnderReview,
    Approved,
    Rejected,
    Escalated,
}

/// Bounty campaign configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BountyCampaign {
    pub campaign_id: Uuid,
    pub name: String,
    pub total_reward_pool: u64,
    pub start_date: i64,
    pub end_date: i64,
    pub merkle_root: Option<[u8; 32]>,
    pub status: CampaignStatus,
}

/// Campaign status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CampaignStatus {
    Pending,
    Active,
    Ended,
    Finalized,
    ClaimWindowOpen,
    Completed,
}

/// Leaderboard entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    pub rank: u32,
    pub testnet_address: [u8; 32],
    pub total_score: f64,
    pub tier: BountyTier,
    pub category_scores: CategoryScores,
}

// ── Tasks / Endeavours ────────────────────────────────────────────────────────

/// A task or endeavour that community members can complete for rewards.
/// Admin-defined (vs bug reports which are user-initiated).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BountyTask {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    /// QUG reward on completion
    pub reward_qug: f64,
    /// Bounty score points awarded
    pub reward_score: f64,
    pub difficulty: TaskDifficulty,
    pub category: TaskCategory,
    pub status: TaskStatus,
    /// None = unlimited completions allowed
    pub max_claims: Option<u32>,
    pub approved_claims: u32,
    /// Unix timestamp deadline, None = no deadline
    pub deadline: Option<i64>,
    pub created_at: i64,
    pub created_by: String,
    /// Human-readable description of what proof to submit
    pub proof_requirements: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskDifficulty {
    Easy,
    Medium,
    Hard,
    Expert,
}

impl TaskDifficulty {
    pub fn score_multiplier(&self) -> f64 {
        match self {
            TaskDifficulty::Easy   => 1.0,
            TaskDifficulty::Medium => 2.0,
            TaskDifficulty::Hard   => 4.0,
            TaskDifficulty::Expert => 8.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskCategory {
    NodeOperation,
    Testing,
    BugHunting,
    Development,
    Documentation,
    Community,
    Security,
    Research,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskStatus {
    Open,
    Closed,
    Archived,
}

/// A user's claim that they have completed a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskClaim {
    pub id: Uuid,
    pub task_id: Uuid,
    pub user_id: Uuid,
    pub wallet_address: String,
    pub proof_url: Option<String>,
    pub proof_text: String,
    pub status: TaskClaimStatus,
    pub submitted_at: i64,
    pub reviewed_at: Option<i64>,
    pub reviewer_notes: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskClaimStatus {
    Pending,
    Approved,
    Rejected,
}
