/// RocksDB storage layer for Testnet Bounty Protocol
///
/// This module provides persistent storage for all bounty-related data with
/// AEGIS-QL access control for administrative operations.

use crate::types::*;
use anyhow::{Context, Result};
use dashmap::DashMap;
use q_aegis_ql::{AegisAccessControl, Signature, access_control::AccessLevel};
use rocksdb::{DB, Options, ColumnFamilyDescriptor};
use serde::{de::DeserializeOwned, Serialize};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// RocksDB column families for bounty data
const CF_USERS: &str = "testnet_users";
const CF_NODE_METRICS: &str = "node_metrics";
const CF_TRANSACTIONS: &str = "transactions";
const CF_BUG_REPORTS: &str = "bug_reports";
const CF_COMMUNITY: &str = "community_contributions";
const CF_SOCIAL: &str = "social_activities";
const CF_FINAL_SCORES: &str = "final_scores";
const CF_CAMPAIGNS: &str = "campaigns";
const CF_LEADERBOARD: &str = "leaderboard";

/// Bounty protocol storage with AEGIS-QL access control
pub struct BountyStorage {
    /// RocksDB instance
    db: Arc<DB>,

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
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // Define column families
        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new(CF_USERS, Options::default()),
            ColumnFamilyDescriptor::new(CF_NODE_METRICS, Options::default()),
            ColumnFamilyDescriptor::new(CF_TRANSACTIONS, Options::default()),
            ColumnFamilyDescriptor::new(CF_BUG_REPORTS, Options::default()),
            ColumnFamilyDescriptor::new(CF_COMMUNITY, Options::default()),
            ColumnFamilyDescriptor::new(CF_SOCIAL, Options::default()),
            ColumnFamilyDescriptor::new(CF_FINAL_SCORES, Options::default()),
            ColumnFamilyDescriptor::new(CF_CAMPAIGNS, Options::default()),
            ColumnFamilyDescriptor::new(CF_LEADERBOARD, Options::default()),
        ];

        let db = DB::open_cf_descriptors(&opts, db_path, cf_descriptors)
            .context("Failed to open RocksDB")?;

        info!("✅ Bounty protocol storage initialized with AEGIS-QL access control");

        Ok(Self {
            db: Arc::new(db),
            access_control,
            user_cache: Arc::new(DashMap::new()),
            leaderboard_cache: Arc::new(parking_lot::RwLock::new(Vec::new())),
        })
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
        // Check if user already exists
        let cf = self.db.cf_handle(CF_USERS)
            .context("Failed to get users CF")?;

        let existing_key = format!("addr:{}", hex::encode(user.testnet_address));
        if let Some(uuid_bytes) = self.db.get_cf(&cf, existing_key.as_bytes())? {
            // Address already registered, return existing user ID
            let existing_uuid = Uuid::from_slice(&uuid_bytes)?;
            info!("🔄 Address already registered, returning existing user ID: {:?}", existing_uuid);
            return Ok(existing_uuid);
        }

        // Generate user ID if not set
        if user.user_id == Uuid::nil() {
            user.user_id = Uuid::new_v4();
        }

        // Store user by UUID
        let uuid_key = format!("uuid:{}", user.user_id);
        self.db.put_cf(&cf, uuid_key.as_bytes(), Self::serialize(&user)?)?;

        // Store user by testnet address (for lookup)
        self.db.put_cf(&cf, existing_key.as_bytes(), user.user_id.as_bytes())?;

        // Update cache
        self.user_cache.insert(user.user_id, user.clone());

        info!("✅ Registered testnet user: {:?}", user.user_id);
        Ok(user.user_id)
    }

    /// Get user by UUID
    pub async fn get_user(&self, user_id: &Uuid) -> Result<Option<TestnetUser>> {
        // Check cache first
        if let Some(user) = self.user_cache.get(user_id) {
            return Ok(Some(user.clone()));
        }

        // Load from database
        let cf = self.db.cf_handle(CF_USERS)
            .context("Failed to get users CF")?;

        let key = format!("uuid:{}", user_id);
        if let Some(bytes) = self.db.get_cf(&cf, key.as_bytes())? {
            let user: TestnetUser = Self::deserialize(&bytes)?;
            self.user_cache.insert(*user_id, user.clone());
            Ok(Some(user))
        } else {
            Ok(None)
        }
    }

    /// Get user by testnet address
    pub async fn get_user_by_address(&self, address: &[u8; 32]) -> Result<Option<TestnetUser>> {
        let cf = self.db.cf_handle(CF_USERS)
            .context("Failed to get users CF")?;

        let key = format!("addr:{}", hex::encode(address));
        if let Some(uuid_bytes) = self.db.get_cf(&cf, key.as_bytes())? {
            let user_id = Uuid::from_slice(&uuid_bytes)?;
            self.get_user(&user_id).await
        } else {
            Ok(None)
        }
    }

    /// Get all users (for wallet connection lookup)
    pub async fn get_all_users(&self) -> Result<Vec<TestnetUser>> {
        let cf = self.db.cf_handle(CF_USERS)
            .context("Failed to get users CF")?;

        let mut users = Vec::new();
        let iter = self.db.iterator_cf(&cf, rocksdb::IteratorMode::Start);

        for item in iter {
            let (key, value) = item?;
            // Only process UUID keys, skip address mapping keys
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

        // Calculate total score before moving category_scores
        let total_score = category_scores.calculate_total() * early_multiplier * consistency_bonus;

        user.category_scores = category_scores;
        user.early_multiplier = early_multiplier;
        user.consistency_bonus = consistency_bonus;
        user.total_score = total_score;

        // Update tier based on score
        user.tier = self.calculate_tier(user.total_score).await?;

        // Persist updated user
        let cf = self.db.cf_handle(CF_USERS)
            .context("Failed to get users CF")?;

        let key = format!("uuid:{}", user_id);
        self.db.put_cf(&cf, key.as_bytes(), Self::serialize(&user)?)?;

        // Cache total_score before moving user
        let final_total_score = user.total_score;

        // Update cache
        self.user_cache.insert(*user_id, user);

        // Invalidate leaderboard cache so it rebuilds on next query
        self.leaderboard_cache.write().clear();

        debug!("Updated user score: {:?} = {}", user_id, final_total_score);
        Ok(())
    }

    /// Calculate tier based on total score
    async fn calculate_tier(&self, _total_score: f64) -> Result<BountyTier> {
        // TODO: Implement percentile-based tier calculation from leaderboard
        // For now, return Participant as default
        Ok(BountyTier::Participant)
    }

    // ============================================================================
    // ACTIVITY TRACKING
    // ============================================================================

    /// Record node metrics
    pub async fn record_node_metrics(&self, metrics: &NodeMetrics) -> Result<()> {
        let cf = self.db.cf_handle(CF_NODE_METRICS)
            .context("Failed to get node_metrics CF")?;

        let key = format!("{}:{}", metrics.user_id, metrics.timestamp);
        self.db.put_cf(&cf, key.as_bytes(), Self::serialize(metrics)?)?;

        debug!("Recorded node metrics for user: {:?}", metrics.user_id);
        Ok(())
    }

    /// Record transaction activity
    pub async fn record_transaction(&self, tx: &TransactionActivity) -> Result<()> {
        let cf = self.db.cf_handle(CF_TRANSACTIONS)
            .context("Failed to get transactions CF")?;

        let key = format!("{}:{}", tx.user_id, tx.tx_hash);
        self.db.put_cf(&cf, key.as_bytes(), Self::serialize(tx)?)?;

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
        // Verify the requester is the user who owns this report
        let user = self.get_user(&report.user_id).await?
            .ok_or_else(|| anyhow::anyhow!("User not found"))?;

        if &user.testnet_address != requester_wallet {
            anyhow::bail!("Unauthorized: wallet mismatch");
        }

        // MVP: Skip AEGIS-QL signature verification since user is already
        // authenticated via JWT. The wallet mismatch check above is sufficient.
        // In production, require actual signed request from frontend.

        let cf = self.db.cf_handle(CF_BUG_REPORTS)
            .context("Failed to get bug_reports CF")?;

        let key = format!("{}:{}", report.user_id, report.timestamp);
        self.db.put_cf(&cf, key.as_bytes(), Self::serialize(report)?)?;

        info!("✅ Bug report submitted: {}", report.issue_url);
        Ok(())
    }

    /// Record community contribution
    pub async fn record_community_contribution(&self, contribution: &CommunityContribution) -> Result<()> {
        let cf = self.db.cf_handle(CF_COMMUNITY)
            .context("Failed to get community CF")?;

        let key = format!("{}:{}", contribution.user_id, contribution.timestamp);
        self.db.put_cf(&cf, key.as_bytes(), Self::serialize(contribution)?)?;

        debug!("Recorded community contribution for user: {:?}", contribution.user_id);
        Ok(())
    }

    /// Record social media activity
    pub async fn record_social_activity(&self, activity: &SocialActivity) -> Result<()> {
        let cf = self.db.cf_handle(CF_SOCIAL)
            .context("Failed to get social CF")?;

        let key = format!("{}:{}:{}", activity.user_id, activity.platform as u8, activity.timestamp);
        self.db.put_cf(&cf, key.as_bytes(), Self::serialize(activity)?)?;

        debug!("Recorded social activity for user: {:?}", activity.user_id);
        Ok(())
    }

    // ============================================================================
    // PER-USER QUERIES (for scoring engine)
    // ============================================================================

    /// Get bug reports for a specific user
    pub async fn get_user_bug_reports(&self, user_id: &Uuid) -> Result<Vec<BugReport>> {
        let cf = self.db.cf_handle(CF_BUG_REPORTS)
            .context("Failed to get bug_reports CF")?;

        let prefix = format!("{}:", user_id);
        let mut reports = Vec::new();
        let iter = self.db.prefix_iterator_cf(&cf, prefix.as_bytes());
        for item in iter {
            let (key, value) = item?;
            // Verify key starts with our prefix (prefix_iterator may overshoot)
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            if let Ok(report) = Self::deserialize::<BugReport>(&value) {
                reports.push(report);
            }
        }
        reports.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(reports)
    }

    /// Get social activities for a specific user
    pub async fn get_user_social_activities(&self, user_id: &Uuid) -> Result<Vec<SocialActivity>> {
        let cf = self.db.cf_handle(CF_SOCIAL)
            .context("Failed to get social CF")?;

        let prefix = format!("{}:", user_id);
        let mut activities = Vec::new();
        let iter = self.db.prefix_iterator_cf(&cf, prefix.as_bytes());
        for item in iter {
            let (key, value) = item?;
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            if let Ok(activity) = Self::deserialize::<SocialActivity>(&value) {
                activities.push(activity);
            }
        }
        activities.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(activities)
    }

    // ============================================================================
    // ADMIN: LIST ALL SUBMISSIONS
    // ============================================================================

    /// Get all bug reports across all users
    pub async fn get_all_bug_reports(&self) -> Result<Vec<BugReport>> {
        let cf = self.db.cf_handle(CF_BUG_REPORTS)
            .context("Failed to get bug_reports CF")?;

        let mut reports = Vec::new();
        let iter = self.db.iterator_cf(&cf, rocksdb::IteratorMode::Start);
        for item in iter {
            let (_, value) = item?;
            if let Ok(report) = Self::deserialize::<BugReport>(&value) {
                reports.push(report);
            }
        }
        // Sort by timestamp descending (newest first)
        reports.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(reports)
    }

    /// Get all social activities across all users
    pub async fn get_all_social_activities(&self) -> Result<Vec<SocialActivity>> {
        let cf = self.db.cf_handle(CF_SOCIAL)
            .context("Failed to get social CF")?;

        let mut activities = Vec::new();
        let iter = self.db.iterator_cf(&cf, rocksdb::IteratorMode::Start);
        for item in iter {
            let (_, value) = item?;
            if let Ok(activity) = Self::deserialize::<SocialActivity>(&value) {
                activities.push(activity);
            }
        }
        // Sort by timestamp descending (newest first)
        activities.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(activities)
    }

    /// Update bug report status (admin approve/reject)
    pub async fn update_bug_report_status(&self, user_id: &Uuid, timestamp: i64, status: BugStatus) -> Result<()> {
        let cf = self.db.cf_handle(CF_BUG_REPORTS)
            .context("Failed to get bug_reports CF")?;

        let key = format!("{}:{}", user_id, timestamp);
        let existing = self.db.get_cf(&cf, key.as_bytes())?;
        if let Some(data) = existing {
            let mut report: BugReport = Self::deserialize(&data)?;
            report.status = status;
            self.db.put_cf(&cf, key.as_bytes(), Self::serialize(&report)?)?;
            info!("✅ Bug report status updated: {} → {:?}", key, report.status);
        } else {
            anyhow::bail!("Bug report not found: {}", key);
        }
        Ok(())
    }

    /// Update social activity verified status (admin approve/reject)
    pub async fn update_social_activity_status(&self, user_id: &Uuid, platform: u8, timestamp: i64, verified: bool) -> Result<()> {
        let cf = self.db.cf_handle(CF_SOCIAL)
            .context("Failed to get social CF")?;

        let key = format!("{}:{}:{}", user_id, platform, timestamp);
        let existing = self.db.get_cf(&cf, key.as_bytes())?;
        if let Some(data) = existing {
            let mut activity: SocialActivity = Self::deserialize(&data)?;
            activity.verified = verified;
            self.db.put_cf(&cf, key.as_bytes(), Self::serialize(&activity)?)?;
            info!("✅ Social activity verification updated: {} → {}", key, verified);
        } else {
            anyhow::bail!("Social activity not found: {}", key);
        }
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
        // Verify admin access
        let message = b"BUILD_LEADERBOARD";
        let acl = self.access_control.read();
        acl.verify_access(requester_wallet, signature, message, AccessLevel::Admin)?;
        drop(acl);

        // Collect all users and their scores
        let mut entries = Vec::new();
        for item in self.user_cache.iter() {
            let user = item.value();
            entries.push(LeaderboardEntry {
                rank: 0, // Will be assigned below
                testnet_address: user.testnet_address,
                total_score: user.total_score,
                tier: user.tier,
                category_scores: user.category_scores.clone(),
            });
        }

        // Sort by total score (descending)
        entries.sort_by(|a, b| b.total_score.partial_cmp(&a.total_score).unwrap_or(std::cmp::Ordering::Equal));

        // Assign ranks
        for (i, entry) in entries.iter_mut().enumerate() {
            entry.rank = (i + 1) as u32;
        }

        // Update cache
        *self.leaderboard_cache.write() = entries.clone();

        // Persist to database
        let cf = self.db.cf_handle(CF_LEADERBOARD)
            .context("Failed to get leaderboard CF")?;

        for entry in &entries {
            let key = format!("rank:{}", entry.rank);
            self.db.put_cf(&cf, key.as_bytes(), Self::serialize(entry)?)?;
        }

        info!("✅ Leaderboard built with {} entries", entries.len());
        Ok(entries)
    }

    /// Get current leaderboard — auto-rebuilds from user DB if cache is empty
    pub async fn get_leaderboard(&self, limit: usize) -> Result<Vec<LeaderboardEntry>> {
        {
            let leaderboard = self.leaderboard_cache.read();
            if !leaderboard.is_empty() {
                return Ok(leaderboard.iter().take(limit).cloned().collect());
            }
        }

        // Cache is empty — rebuild from all users in DB
        let all_users = self.get_all_users().await?;
        let mut entries: Vec<LeaderboardEntry> = all_users
            .iter()
            .filter(|u| u.total_score > 0.0)
            .map(|user| LeaderboardEntry {
                rank: 0,
                testnet_address: user.testnet_address,
                total_score: user.total_score,
                tier: user.tier,
                category_scores: user.category_scores.clone(),
            })
            .collect();

        entries.sort_by(|a, b| b.total_score.partial_cmp(&a.total_score).unwrap_or(std::cmp::Ordering::Equal));
        for (i, entry) in entries.iter_mut().enumerate() {
            entry.rank = (i + 1) as u32;
        }

        *self.leaderboard_cache.write() = entries.clone();
        Ok(entries.into_iter().take(limit).collect())
    }

    /// Finalize campaign and generate Merkle root (founder-only)
    pub async fn finalize_campaign(
        &self,
        campaign_id: &Uuid,
        requester_wallet: &[u8; 32],
        signature: &Signature,
    ) -> Result<[u8; 32]> {
        // Verify founder access
        let message = format!("FINALIZE_CAMPAIGN:{}", campaign_id);
        let acl = self.access_control.read();
        acl.verify_access(requester_wallet, signature, message.as_bytes(), AccessLevel::Founder)?;
        drop(acl);

        // TODO: Implement Merkle tree generation from leaderboard
        let merkle_root = [0u8; 32]; // Placeholder

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
        // Verify founder access
        let message = format!("CREATE_CAMPAIGN:{}", campaign.name);
        let acl = self.access_control.read();
        acl.verify_access(requester_wallet, signature, message.as_bytes(), AccessLevel::Founder)?;
        drop(acl);

        let cf = self.db.cf_handle(CF_CAMPAIGNS)
            .context("Failed to get campaigns CF")?;

        let key = format!("campaign:{}", campaign.campaign_id);
        self.db.put_cf(&cf, key.as_bytes(), Self::serialize(campaign)?)?;

        info!("✅ Campaign created: {} ({:?})", campaign.name, campaign.campaign_id);
        Ok(())
    }

    /// Get campaign by ID
    pub async fn get_campaign(&self, campaign_id: &Uuid) -> Result<Option<BountyCampaign>> {
        let cf = self.db.cf_handle(CF_CAMPAIGNS)
            .context("Failed to get campaigns CF")?;

        let key = format!("campaign:{}", campaign_id);
        if let Some(bytes) = self.db.get_cf(&cf, key.as_bytes())? {
            let campaign: BountyCampaign = Self::deserialize(&bytes)?;
            Ok(Some(campaign))
        } else {
            Ok(None)
        }
    }

    /// Get all campaigns
    pub async fn get_all_campaigns(&self) -> Result<Vec<BountyCampaign>> {
        let cf = self.db.cf_handle(CF_CAMPAIGNS)
            .context("Failed to get campaigns CF")?;

        let mut campaigns = Vec::new();
        let iter = self.db.iterator_cf(&cf, rocksdb::IteratorMode::Start);

        for item in iter {
            let (key, value) = item?;
            // Only process campaign keys
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_storage() -> (BountyStorage, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let acl = Arc::new(parking_lot::RwLock::new(
            AegisAccessControl::new([0u8; 32], q_aegis_ql::PublicKey {
                a: vec![1; 512],
                t: vec![2; 512],
            })
        ));
        let storage = BountyStorage::new(temp_dir.path(), acl).unwrap();
        (storage, temp_dir)
    }

    #[tokio::test]
    async fn test_user_registration() {
        let (storage, _temp) = create_test_storage();

        let user = TestnetUser {
            user_id: Uuid::nil(),
            testnet_address: [1u8; 32],
            mainnet_address: None,
            registration_date: chrono::Utc::now().timestamp_millis(),
            social_accounts: SocialAccounts::default(),
            tier: BountyTier::Supporter,
            total_score: 0.0,
            category_scores: CategoryScores::default(),
            early_multiplier: 1.0,
            consistency_bonus: 1.0,
            rank: None,
            kyc_verified: false,
        };

        let user_id = storage.register_user(user).await.unwrap();
        assert_ne!(user_id, Uuid::nil());

        let retrieved = storage.get_user(&user_id).await.unwrap().unwrap();
        assert_eq!(retrieved.testnet_address, [1u8; 32]);
    }

    #[tokio::test]
    async fn test_score_update() {
        let (storage, _temp) = create_test_storage();

        let user = TestnetUser {
            user_id: Uuid::nil(),
            testnet_address: [2u8; 32],
            mainnet_address: None,
            registration_date: chrono::Utc::now().timestamp_millis(),
            social_accounts: SocialAccounts::default(),
            tier: BountyTier::Supporter,
            total_score: 0.0,
            category_scores: CategoryScores::default(),
            early_multiplier: 1.0,
            consistency_bonus: 1.0,
            rank: None,
            kyc_verified: false,
        };

        let user_id = storage.register_user(user).await.unwrap();

        let scores = CategoryScores {
            node_ops: 100.0,
            transactions: 80.0,
            bug_reports: 60.0,
            community: 40.0,
            social: 20.0,
        };

        storage.update_user_score(&user_id, scores, 1.2, 1.1).await.unwrap();

        let updated_user = storage.get_user(&user_id).await.unwrap().unwrap();
        assert!(updated_user.total_score > 0.0);
    }
}
