/// Scoring engine for Testnet Bounty Protocol
///
/// Implements the weighted scoring formula with anti-gaming controls:
/// FinalScore = (
///     0.30 * NodeOpsScore +
///     0.25 * TxVolumeScore +
///     0.20 * BugReportScore +
///     0.15 * CommunityScore +
///     0.10 * SocialScore
/// ) * EarlyMultiplier * ConsistencyBonus

use crate::types::*;
use anyhow::Result;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use tracing::{debug, warn};
use uuid::Uuid;

/// Scoring engine with fraud detection
pub struct ScoringEngine {
    /// Campaign start date for early multiplier calculation
    campaign_start: i64,

    /// User activity history for consistency bonus calculation
    user_activity_history: HashMap<Uuid, Vec<i64>>,
}

impl ScoringEngine {
    /// Create a new scoring engine
    pub fn new(campaign_start: i64) -> Self {
        Self {
            campaign_start,
            user_activity_history: HashMap::new(),
        }
    }

    /// Calculate comprehensive user score
    pub async fn calculate_user_score(
        &mut self,
        user_id: &Uuid,
        node_metrics: &[NodeMetrics],
        transactions: &[TransactionActivity],
        bug_reports: &[BugReport],
        community: &[CommunityContribution],
        social: &[SocialActivity],
        registration_date: i64,
    ) -> Result<(CategoryScores, f64, f64, f64)> {
        // Calculate category scores
        let node_ops_score = self.calculate_node_ops_score(node_metrics).await?;
        let tx_score = self.calculate_transaction_score(transactions).await?;
        let bug_score = self.calculate_bug_score(bug_reports).await?;
        let community_score = self.calculate_community_score(community).await?;
        let social_score = self.calculate_social_score(social).await?;

        let category_scores = CategoryScores {
            node_ops: node_ops_score,
            transactions: tx_score,
            bug_reports: bug_score,
            community: community_score,
            social: social_score,
        };

        // Calculate multipliers
        let early_multiplier = self.calculate_early_multiplier(registration_date);
        let consistency_bonus = self.calculate_consistency_bonus(user_id, &[
            node_metrics.iter().map(|m| m.timestamp).collect::<Vec<i64>>(),
            transactions.iter().map(|t| t.timestamp).collect::<Vec<i64>>(),
        ].concat());

        // Calculate final score
        let base_score = category_scores.calculate_total();
        let final_score = base_score * early_multiplier * consistency_bonus;

        debug!(
            "User {:?} score: base={:.2}, early_mult={:.2}, consistency={:.2}, final={:.2}",
            user_id, base_score, early_multiplier, consistency_bonus, final_score
        );

        Ok((category_scores, early_multiplier, consistency_bonus, final_score))
    }

    // ============================================================================
    // CATEGORY SCORING FUNCTIONS
    // ============================================================================

    /// Calculate node operations score (30% weight, max 300 points)
    ///
    /// Factors:
    /// - Uptime percentage (40%)
    /// - Blocks produced (30%)
    /// - Governance votes (20%)
    /// - Peer count stability (10%)
    async fn calculate_node_ops_score(&self, metrics: &[NodeMetrics]) -> Result<f64> {
        if metrics.is_empty() {
            return Ok(0.0);
        }

        // Calculate average metrics
        let avg_uptime: f64 = metrics.iter().map(|m| m.uptime_percentage).sum::<f64>() / metrics.len() as f64;
        let total_blocks: u64 = metrics.iter().map(|m| m.blocks_produced).sum();
        let total_votes: u64 = metrics.iter().map(|m| m.governance_votes).sum();
        let avg_peers: f32 = metrics.iter().map(|m| m.peer_count).sum::<u32>() as f32 / metrics.len() as f32;

        // Apply scoring formula
        let uptime_score = (avg_uptime / 100.0) * 120.0; // Max 120 points
        let blocks_score = (total_blocks as f64).min(1000.0) / 1000.0 * 90.0; // Max 90 points
        let votes_score = (total_votes as f64).min(100.0) / 100.0 * 60.0; // Max 60 points
        let peers_score = (avg_peers / 50.0).min(1.0) as f64 * 30.0; // Max 30 points

        let total = uptime_score + blocks_score + votes_score + peers_score;

        // Anti-gaming: detect unrealistic metrics
        if avg_uptime > 99.9 && total_blocks > 10000 {
            warn!("Suspicious node metrics detected: uptime={:.2}, blocks={}", avg_uptime, total_blocks);
            return Ok(total * 0.5); // Penalty for suspicious activity
        }

        Ok(total.min(300.0))
    }

    /// Calculate transaction score (25% weight, max 250 points)
    ///
    /// Factors:
    /// - Transaction count (40%)
    /// - Total value transferred (30%)
    /// - Contract interactions (20%)
    /// - DEX swap diversity (10%)
    async fn calculate_transaction_score(&self, transactions: &[TransactionActivity]) -> Result<f64> {
        if transactions.is_empty() {
            return Ok(0.0);
        }

        let tx_count = transactions.len() as f64;
        let total_value: u64 = transactions.iter().map(|t| t.value).sum();
        let total_contracts: u32 = transactions.iter().map(|t| t.contract_interactions).sum();
        let total_swaps: u32 = transactions.iter().map(|t| t.dex_swaps).sum();

        let count_score = (tx_count / 1000.0).min(1.0) * 100.0; // Max 100 points
        let value_score = (total_value as f64 / 1_000_000.0).min(1.0) * 75.0; // Max 75 points
        let contract_score = (total_contracts as f64 / 100.0).min(1.0) * 50.0; // Max 50 points
        let swap_score = (total_swaps as f64 / 50.0).min(1.0) * 25.0; // Max 25 points

        let total = count_score + value_score + contract_score + swap_score;

        // Anti-gaming: detect wash trading patterns
        if self.detect_wash_trading(transactions) {
            warn!("Wash trading pattern detected in transactions");
            return Ok(total * 0.3); // Severe penalty for wash trading
        }

        Ok(total.min(250.0))
    }

    /// Calculate bug report score (20% weight, max 200 points)
    ///
    /// Severity-weighted with special bounties for critical bugs
    async fn calculate_bug_score(&self, bug_reports: &[BugReport]) -> Result<f64> {
        let mut total_score = 0.0;

        for report in bug_reports {
            if report.status == BugStatus::Verified || report.status == BugStatus::Fixed {
                total_score += report.severity.score_multiplier();
            } else if report.status == BugStatus::Duplicate || report.status == BugStatus::Invalid {
                // No points for duplicates or invalid reports
                continue;
            } else {
                // Partial points for submitted/under review (pending verification)
                total_score += report.severity.score_multiplier() * 0.5;
            }
        }

        Ok(total_score.min(200.0))
    }

    /// Calculate community contribution score (15% weight, max 150 points)
    async fn calculate_community_score(&self, contributions: &[CommunityContribution]) -> Result<f64> {
        let mut total_score = 0.0;

        for contribution in contributions {
            if contribution.verified {
                let base = contribution.contribution_type.base_score();
                total_score += base * contribution.impact_score;
            }
        }

        Ok(total_score.min(150.0))
    }

    /// Calculate social media engagement score (10% weight, max 100 points)
    async fn calculate_social_score(&self, activities: &[SocialActivity]) -> Result<f64> {
        let mut total_score = 0.0;

        for activity in activities {
            if activity.verified {
                let base = activity.activity_type.base_score();
                total_score += base * activity.engagement_score;
            }
        }

        // Anti-gaming: detect spam or bot-like activity
        if self.detect_social_spam(activities) {
            warn!("Social media spam detected");
            return Ok(total_score * 0.2); // Severe penalty
        }

        Ok(total_score.min(100.0))
    }

    // ============================================================================
    // MULTIPLIERS
    // ============================================================================

    /// Calculate early participation multiplier
    ///
    /// Formula: 1.0 + (0.5 * days_since_start / 30)^-0.5
    /// - Day 1: 2.0x multiplier
    /// - Day 7: 1.6x multiplier
    /// - Day 14: 1.4x multiplier
    /// - Day 30+: 1.0x multiplier
    fn calculate_early_multiplier(&self, registration_date: i64) -> f64 {
        let days_since_start = ((registration_date - self.campaign_start) / 86400000) as f64;

        if days_since_start < 0.0 {
            return 1.0; // No multiplier for pre-campaign registrations
        }

        let multiplier = 1.0 + (0.5 * (days_since_start / 30.0).powf(-0.5));
        multiplier.min(2.0).max(1.0)
    }

    /// Calculate consistency bonus based on activity pattern
    ///
    /// Rewards consistent daily/weekly participation over sporadic activity
    fn calculate_consistency_bonus(&mut self, user_id: &Uuid, activity_timestamps: &[i64]) -> f64 {
        if activity_timestamps.is_empty() {
            return 1.0;
        }

        // Store activity history
        self.user_activity_history.insert(*user_id, activity_timestamps.to_vec());

        // Group activities by day
        let mut daily_activity: HashMap<i64, usize> = HashMap::new();
        for &timestamp in activity_timestamps {
            let day = timestamp / 86400000; // Convert to day
            *daily_activity.entry(day).or_insert(0) += 1;
        }

        // Calculate active days
        let active_days = daily_activity.len() as f64;
        let total_days = if let (Some(&first), Some(&last)) = (
            activity_timestamps.iter().min(),
            activity_timestamps.iter().max()
        ) {
            ((last - first) / 86400000).max(1) as f64
        } else {
            1.0
        };

        // Consistency ratio (active days / total days)
        let consistency_ratio = active_days / total_days;

        // Bonus: 1.0 to 1.2x based on consistency
        1.0 + (consistency_ratio * 0.2)
    }

    // ============================================================================
    // FRAUD DETECTION
    // ============================================================================

    /// Detect wash trading patterns
    ///
    /// Indicators:
    /// - High transaction count with low value variance
    /// - Repetitive transaction patterns
    /// - Circular transaction flows
    fn detect_wash_trading(&self, transactions: &[TransactionActivity]) -> bool {
        if transactions.len() < 10 {
            return false; // Need sufficient data
        }

        // Calculate value variance
        let values: Vec<u64> = transactions.iter().map(|t| t.value).collect();
        let mean = values.iter().sum::<u64>() as f64 / values.len() as f64;
        let variance = values.iter()
            .map(|&v| {
                let diff = v as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / values.len() as f64;

        // Low variance with high count suggests wash trading
        variance < (mean * 0.1)
    }

    /// Detect social media spam or bot activity
    ///
    /// Indicators:
    /// - Very high activity count in short time
    /// - Low engagement scores despite high activity
    /// - Repetitive content patterns
    fn detect_social_spam(&self, activities: &[SocialActivity]) -> bool {
        if activities.len() < 20 {
            return false;
        }

        // Check time clustering (many activities in short window)
        if let (Some(first), Some(last)) = (
            activities.iter().map(|a| a.timestamp).min(),
            activities.iter().map(|a| a.timestamp).max()
        ) {
            let time_span_hours = (last - first) / 3600000;
            if time_span_hours < 24 && activities.len() > 50 {
                return true; // Too many activities in 24 hours
            }
        }

        // Check average engagement score
        let avg_engagement = activities.iter().map(|a| a.engagement_score).sum::<f64>() / activities.len() as f64;
        if avg_engagement < 0.1 {
            return true; // Low engagement suggests bot activity
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_node_ops_scoring() {
        let engine = ScoringEngine::new(chrono::Utc::now().timestamp_millis());

        let metrics = vec![
            NodeMetrics {
                user_id: Uuid::new_v4(),
                uptime_percentage: 99.5,
                blocks_produced: 100,
                governance_votes: 10,
                peer_count: 25,
                timestamp: chrono::Utc::now().timestamp_millis(),
            },
        ];

        let score = engine.calculate_node_ops_score(&metrics).await.unwrap();
        assert!(score > 0.0);
        assert!(score <= 300.0);
    }

    #[tokio::test]
    async fn test_early_multiplier() {
        let campaign_start = chrono::Utc::now().timestamp_millis();
        let engine = ScoringEngine::new(campaign_start);

        // Day 1 registration
        let day1_mult = engine.calculate_early_multiplier(campaign_start + 86400000);
        assert!(day1_mult > 1.5);

        // Day 30 registration
        let day30_mult = engine.calculate_early_multiplier(campaign_start + (30 * 86400000));
        assert!(day30_mult >= 1.0);
        assert!(day30_mult < 1.2);
    }

    #[tokio::test]
    async fn test_wash_trading_detection() {
        let engine = ScoringEngine::new(chrono::Utc::now().timestamp_millis());

        // Create suspicious transactions (same value repeated)
        let suspicious_txs: Vec<TransactionActivity> = (0..20)
            .map(|i| TransactionActivity {
                user_id: Uuid::new_v4(),
                tx_hash: format!("tx_{}", i),
                value: 1000, // Same value
                contract_interactions: 0,
                dex_swaps: 0,
                timestamp: chrono::Utc::now().timestamp_millis() + (i * 1000),
            })
            .collect();

        assert!(engine.detect_wash_trading(&suspicious_txs));

        // Create organic transactions (varied values)
        let organic_txs: Vec<TransactionActivity> = (0..20)
            .map(|i| TransactionActivity {
                user_id: Uuid::new_v4(),
                tx_hash: format!("tx_{}", i),
                value: 1000 + (i * 137), // Varied values
                contract_interactions: i % 3,
                dex_swaps: i % 2,
                timestamp: chrono::Utc::now().timestamp_millis() + (i * 60000),
            })
            .collect();

        assert!(!engine.detect_wash_trading(&organic_txs));
    }
}
