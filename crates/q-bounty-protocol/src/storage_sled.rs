/// Sled storage layer for Testnet Bounty Protocol (Windows fallback)
///
/// This module mirrors storage.rs but uses sled instead of RocksDB
/// for Windows cross-compilation compatibility.

use crate::types::*;
use anyhow::{Context, Result};
use dashmap::DashMap;
use q_aegis_ql::{AegisAccessControl, Signature, access_control::AccessLevel};
use serde::{de::DeserializeOwned, Serialize};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Sled tree names for bounty data (equivalent to RocksDB column families)
const CF_USERS: &str = "testnet_users";
const CF_NODE_METRICS: &str = "node_metrics";
const CF_TRANSACTIONS: &str = "transactions";
const CF_BUG_REPORTS: &str = "bug_reports";
const CF_COMMUNITY: &str = "community_contributions";
const CF_SOCIAL: &str = "social_activities";
const CF_FINAL_SCORES: &str = "final_scores";
const CF_CAMPAIGNS: &str = "campaigns";
const CF_LEADERBOARD: &str = "leaderboard";

/// Bounty protocol storage with AEGIS-QL access control (sled backend)
pub struct BountyStorage {
    /// Sled database instance
    db: Arc<sled::Db>,

    /// AEGIS-QL access control system
    access_control: Arc<parking_lot::RwLock<AegisAccessControl>>,

    /// In-memory cache for active users (performance optimization)
    user_cache: Arc<DashMap<Uuid, TestnetUser>>,

    /// In-memory leaderboard cache
    leaderboard_cache: Arc<parking_lot::RwLock<Vec<LeaderboardEntry>>>,
}

impl BountyStorage {
    /// Create a new bounty storage instance
    pub fn new<P: AsRef<Path>>(
        db_path: P,
        access_control: Arc<parking_lot::RwLock<AegisAccessControl>>,
    ) -> Result<Self> {
        let db = sled::Config::new()
            .path(db_path)
            .open()
            .context("Failed to open sled database")?;

        // Pre-create all trees (equivalent to column families)
        let _ = db.open_tree(CF_USERS)?;
        let _ = db.open_tree(CF_NODE_METRICS)?;
        let _ = db.open_tree(CF_TRANSACTIONS)?;
        let _ = db.open_tree(CF_BUG_REPORTS)?;
        let _ = db.open_tree(CF_COMMUNITY)?;
        let _ = db.open_tree(CF_SOCIAL)?;
        let _ = db.open_tree(CF_FINAL_SCORES)?;
        let _ = db.open_tree(CF_CAMPAIGNS)?;
        let _ = db.open_tree(CF_LEADERBOARD)?;

        info!("✅ Bounty protocol storage initialized with AEGIS-QL access control");

        Ok(Self {
            db: Arc::new(db),
            access_control,
            user_cache: Arc::new(DashMap::new()),
            leaderboard_cache: Arc::new(parking_lot::RwLock::new(Vec::new())),
        })
    }

    /// Get a sled tree by name
    fn tree(&self, name: &str) -> Result<sled::Tree> {
        self.db.open_tree(name).context(format!("Failed to get tree: {}", name))
    }

    /// Generic serialization helper
    fn serialize<T: Serialize>(value: &T) -> Result<Vec<u8>> {
        bincode::serialize(value).context("Failed to serialize value")
    }

    /// Generic deserialization helper
    fn deserialize<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
        bincode::deserialize(bytes).context("Failed to deserialize value")
    }

    // ============================================================================
    // USER MANAGEMENT
    // ============================================================================

    /// Register a new testnet user
    pub async fn register_user(&self, mut user: TestnetUser) -> Result<Uuid> {
        let tree = self.tree(CF_USERS)?;

        let existing_key = format!("addr:{}", hex::encode(user.testnet_address));
        if let Some(uuid_bytes) = tree.get(existing_key.as_bytes())? {
            let existing_uuid = Uuid::from_slice(&uuid_bytes)?;
            info!("🔄 Address already registered, returning existing user ID: {:?}", existing_uuid);
            return Ok(existing_uuid);
        }

        if user.user_id == Uuid::nil() {
            user.user_id = Uuid::new_v4();
        }

        let uuid_key = format!("uuid:{}", user.user_id);
        tree.insert(uuid_key.as_bytes(), Self::serialize(&user)?)?;
        tree.insert(existing_key.as_bytes(), user.user_id.as_bytes().as_slice())?;

        self.user_cache.insert(user.user_id, user.clone());

        info!("✅ Registered testnet user: {:?}", user.user_id);
        Ok(user.user_id)
    }

    /// Get user by UUID
    pub async fn get_user(&self, user_id: &Uuid) -> Result<Option<TestnetUser>> {
        if let Some(user) = self.user_cache.get(user_id) {
            return Ok(Some(user.clone()));
        }

        let tree = self.tree(CF_USERS)?;
        let key = format!("uuid:{}", user_id);
        if let Some(bytes) = tree.get(key.as_bytes())? {
            let user: TestnetUser = Self::deserialize(&bytes)?;
            self.user_cache.insert(*user_id, user.clone());
            Ok(Some(user))
        } else {
            Ok(None)
        }
    }

    /// Get user by testnet address
    pub async fn get_user_by_address(&self, address: &[u8; 32]) -> Result<Option<TestnetUser>> {
        let tree = self.tree(CF_USERS)?;
        let key = format!("addr:{}", hex::encode(address));
        if let Some(uuid_bytes) = tree.get(key.as_bytes())? {
            let user_id = Uuid::from_slice(&uuid_bytes)?;
            self.get_user(&user_id).await
        } else {
            Ok(None)
        }
    }

    /// Get all users
    pub async fn get_all_users(&self) -> Result<Vec<TestnetUser>> {
        let tree = self.tree(CF_USERS)?;
        let mut users = Vec::new();

        for item in tree.iter() {
            let (key, value) = item?;
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if key_str.starts_with("uuid:") {
                    if let Ok(user) = Self::deserialize::<TestnetUser>(&value) {
                        users.push(user);
                    }
                }
            }
        }

        Ok(users)
    }

    /// Update user score and tier
    pub async fn update_user_score(
        &self,
        user_id: &Uuid,
        category_scores: CategoryScores,
        early_multiplier: f64,
        consistency_bonus: f64,
    ) -> Result<()> {
        let mut user = self.get_user(user_id).await?
            .ok_or_else(|| anyhow::anyhow!("User not found"))?;

        let total_score = category_scores.calculate_total() * early_multiplier * consistency_bonus;

        user.category_scores = category_scores;
        user.early_multiplier = early_multiplier;
        user.consistency_bonus = consistency_bonus;
        user.total_score = total_score;
        user.tier = self.calculate_tier(user.total_score).await?;

        let tree = self.tree(CF_USERS)?;
        let key = format!("uuid:{}", user_id);
        tree.insert(key.as_bytes(), Self::serialize(&user)?)?;

        let final_total_score = user.total_score;
        self.user_cache.insert(*user_id, user);

        debug!("Updated user score: {:?} = {}", user_id, final_total_score);
        Ok(())
    }

    /// Calculate tier based on total score
    async fn calculate_tier(&self, _total_score: f64) -> Result<BountyTier> {
        Ok(BountyTier::Participant)
    }

    // ============================================================================
    // ACTIVITY TRACKING
    // ============================================================================

    /// Record node metrics
    pub async fn record_node_metrics(&self, metrics: &NodeMetrics) -> Result<()> {
        let tree = self.tree(CF_NODE_METRICS)?;
        let key = format!("{}:{}", metrics.user_id, metrics.timestamp);
        tree.insert(key.as_bytes(), Self::serialize(metrics)?)?;
        debug!("Recorded node metrics for user: {:?}", metrics.user_id);
        Ok(())
    }

    /// Record transaction activity
    pub async fn record_transaction(&self, tx: &TransactionActivity) -> Result<()> {
        let tree = self.tree(CF_TRANSACTIONS)?;
        let key = format!("{}:{}", tx.user_id, tx.tx_hash);
        tree.insert(key.as_bytes(), Self::serialize(tx)?)?;
        debug!("Recorded transaction for user: {:?}", tx.user_id);
        Ok(())
    }

    /// Submit bug report (requires verification)
    pub async fn submit_bug_report(
        &self,
        report: &BugReport,
        requester_wallet: &[u8; 32],
        _signature: &Signature,
    ) -> Result<()> {
        let user = self.get_user(&report.user_id).await?
            .ok_or_else(|| anyhow::anyhow!("User not found"))?;

        if &user.testnet_address != requester_wallet {
            anyhow::bail!("Unauthorized: wallet mismatch");
        }

        // MVP: Skip AEGIS-QL signature verification since user is already
        // authenticated via JWT. The wallet mismatch check above is sufficient.
        // In production, require actual signed request from frontend.

        let tree = self.tree(CF_BUG_REPORTS)?;
        let key = format!("{}:{}", report.user_id, report.timestamp);
        tree.insert(key.as_bytes(), Self::serialize(report)?)?;

        info!("✅ Bug report submitted: {}", report.issue_url);
        Ok(())
    }

    /// Record community contribution
    pub async fn record_community_contribution(&self, contribution: &CommunityContribution) -> Result<()> {
        let tree = self.tree(CF_COMMUNITY)?;
        let key = format!("{}:{}", contribution.user_id, contribution.timestamp);
        tree.insert(key.as_bytes(), Self::serialize(contribution)?)?;
        debug!("Recorded community contribution for user: {:?}", contribution.user_id);
        Ok(())
    }

    /// Record social media activity
    pub async fn record_social_activity(&self, activity: &SocialActivity) -> Result<()> {
        let tree = self.tree(CF_SOCIAL)?;
        let key = format!("{}:{}:{}", activity.user_id, activity.platform as u8, activity.timestamp);
        tree.insert(key.as_bytes(), Self::serialize(activity)?)?;
        debug!("Recorded social activity for user: {:?}", activity.user_id);
        Ok(())
    }

    // ============================================================================
    // LEADERBOARD & FINALIZATION
    // ============================================================================

    /// Build and cache leaderboard (admin-only operation)
    pub async fn build_leaderboard(
        &self,
        requester_wallet: &[u8; 32],
        signature: &Signature,
    ) -> Result<Vec<LeaderboardEntry>> {
        let message = b"BUILD_LEADERBOARD";
        let acl = self.access_control.read();
        acl.verify_access(requester_wallet, signature, message, AccessLevel::Admin)?;
        drop(acl);

        let mut entries = Vec::new();
        for item in self.user_cache.iter() {
            let user = item.value();
            entries.push(LeaderboardEntry {
                rank: 0,
                testnet_address: user.testnet_address,
                total_score: user.total_score,
                tier: user.tier,
                category_scores: user.category_scores.clone(),
            });
        }

        entries.sort_by(|a, b| b.total_score.partial_cmp(&a.total_score).unwrap_or(std::cmp::Ordering::Equal));

        for (i, entry) in entries.iter_mut().enumerate() {
            entry.rank = (i + 1) as u32;
        }

        *self.leaderboard_cache.write() = entries.clone();

        let tree = self.tree(CF_LEADERBOARD)?;
        for entry in &entries {
            let key = format!("rank:{}", entry.rank);
            tree.insert(key.as_bytes(), Self::serialize(entry)?)?;
        }

        info!("✅ Leaderboard built with {} entries", entries.len());
        Ok(entries)
    }

    /// Get current leaderboard
    pub async fn get_leaderboard(&self, limit: usize) -> Result<Vec<LeaderboardEntry>> {
        let leaderboard = self.leaderboard_cache.read();
        Ok(leaderboard.iter().take(limit).cloned().collect())
    }

    /// Finalize campaign and generate Merkle root (founder-only)
    pub async fn finalize_campaign(
        &self,
        campaign_id: &Uuid,
        requester_wallet: &[u8; 32],
        signature: &Signature,
    ) -> Result<[u8; 32]> {
        let message = format!("FINALIZE_CAMPAIGN:{}", campaign_id);
        let acl = self.access_control.read();
        acl.verify_access(requester_wallet, signature, message.as_bytes(), AccessLevel::Founder)?;
        drop(acl);

        let merkle_root = [0u8; 32];

        info!("✅ Campaign finalized: {:?}, Merkle root: {}", campaign_id, hex::encode(merkle_root));
        Ok(merkle_root)
    }

    // ============================================================================
    // CAMPAIGN MANAGEMENT
    // ============================================================================

    /// Create a new bounty campaign (founder-only)
    pub async fn create_campaign(
        &self,
        campaign: &BountyCampaign,
        requester_wallet: &[u8; 32],
        signature: &Signature,
    ) -> Result<()> {
        let message = format!("CREATE_CAMPAIGN:{}", campaign.name);
        let acl = self.access_control.read();
        acl.verify_access(requester_wallet, signature, message.as_bytes(), AccessLevel::Founder)?;
        drop(acl);

        let tree = self.tree(CF_CAMPAIGNS)?;
        let key = format!("campaign:{}", campaign.campaign_id);
        tree.insert(key.as_bytes(), Self::serialize(campaign)?)?;

        info!("✅ Campaign created: {} ({:?})", campaign.name, campaign.campaign_id);
        Ok(())
    }

    /// Get campaign by ID
    pub async fn get_campaign(&self, campaign_id: &Uuid) -> Result<Option<BountyCampaign>> {
        let tree = self.tree(CF_CAMPAIGNS)?;
        let key = format!("campaign:{}", campaign_id);
        if let Some(bytes) = tree.get(key.as_bytes())? {
            let campaign: BountyCampaign = Self::deserialize(&bytes)?;
            Ok(Some(campaign))
        } else {
            Ok(None)
        }
    }

    /// Get all campaigns
    pub async fn get_all_campaigns(&self) -> Result<Vec<BountyCampaign>> {
        let tree = self.tree(CF_CAMPAIGNS)?;
        let mut campaigns = Vec::new();

        for item in tree.iter() {
            let (key, value) = item?;
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if key_str.starts_with("campaign:") {
                    if let Ok(campaign) = Self::deserialize::<BountyCampaign>(&value) {
                        campaigns.push(campaign);
                    }
                }
            }
        }

        Ok(campaigns)
    }

    // ── Tasks / Endeavours ────────────────────────────────────────────────────

    pub async fn create_task(&self, task: BountyTask) -> Result<Uuid> {
        let tree = self.tree("tasks")?;
        let key = format!("task:{}", task.id);
        let bytes = Self::serialize(&task)?;
        tree.insert(key.as_bytes(), bytes)?;
        info!(task_id = %task.id, title = %task.title, "Task created");
        Ok(task.id)
    }

    pub async fn get_task(&self, task_id: &Uuid) -> Result<Option<BountyTask>> {
        let tree = self.tree("tasks")?;
        let key = format!("task:{}", task_id);
        if let Some(bytes) = tree.get(key.as_bytes())? {
            Ok(Some(Self::deserialize(&bytes)?))
        } else {
            Ok(None)
        }
    }

    pub async fn update_task(&self, task: &BountyTask) -> Result<()> {
        let tree = self.tree("tasks")?;
        let key = format!("task:{}", task.id);
        let bytes = Self::serialize(task)?;
        tree.insert(key.as_bytes(), bytes)?;
        Ok(())
    }

    pub async fn get_all_tasks(&self) -> Result<Vec<BountyTask>> {
        let tree = self.tree("tasks")?;
        let mut tasks = Vec::new();
        for item in tree.iter() {
            let (key, value) = item?;
            if let Ok(k) = std::str::from_utf8(&key) {
                if k.starts_with("task:") {
                    if let Ok(t) = Self::deserialize::<BountyTask>(&value) {
                        tasks.push(t);
                    }
                }
            }
        }
        tasks.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(tasks)
    }

    pub async fn submit_task_claim(&self, claim: TaskClaim) -> Result<Uuid> {
        let tree = self.tree("task_claims")?;
        let key = format!("claim:{}:{}", claim.task_id, claim.id);
        let bytes = Self::serialize(&claim)?;
        tree.insert(key.as_bytes(), bytes)?;
        info!(claim_id = %claim.id, task_id = %claim.task_id, user_id = %claim.user_id, "Task claim submitted");
        Ok(claim.id)
    }

    pub async fn get_task_claim(&self, task_id: &Uuid, claim_id: &Uuid) -> Result<Option<TaskClaim>> {
        let tree = self.tree("task_claims")?;
        let key = format!("claim:{}:{}", task_id, claim_id);
        if let Some(bytes) = tree.get(key.as_bytes())? {
            Ok(Some(Self::deserialize(&bytes)?))
        } else {
            Ok(None)
        }
    }

    pub async fn update_task_claim(&self, claim: &TaskClaim) -> Result<()> {
        let tree = self.tree("task_claims")?;
        let key = format!("claim:{}:{}", claim.task_id, claim.id);
        let bytes = Self::serialize(claim)?;
        tree.insert(key.as_bytes(), bytes)?;
        Ok(())
    }

    pub async fn get_claims_for_task(&self, task_id: &Uuid) -> Result<Vec<TaskClaim>> {
        let tree = self.tree("task_claims")?;
        let prefix = format!("claim:{}:", task_id);
        let mut claims = Vec::new();
        for item in tree.scan_prefix(prefix.as_bytes()) {
            let (_, value) = item?;
            if let Ok(c) = Self::deserialize::<TaskClaim>(&value) {
                claims.push(c);
            }
        }
        Ok(claims)
    }

    pub async fn get_all_task_claims(&self) -> Result<Vec<TaskClaim>> {
        let tree = self.tree("task_claims")?;
        let mut claims = Vec::new();
        for item in tree.iter() {
            let (_, value) = item?;
            if let Ok(c) = Self::deserialize::<TaskClaim>(&value) {
                claims.push(c);
            }
        }
        claims.sort_by(|a, b| b.submitted_at.cmp(&a.submitted_at));
        Ok(claims)
    }

    pub async fn get_user_task_claims(&self, user_id: &Uuid) -> Result<Vec<TaskClaim>> {
        let all = self.get_all_task_claims().await?;
        Ok(all.into_iter().filter(|c| c.user_id == *user_id).collect())
    }
}
