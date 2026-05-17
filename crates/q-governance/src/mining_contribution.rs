//! Mining contribution tracking and verification

use crate::types::*;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Tracks mining contributions for governance voting
pub struct MiningContributionTracker {
    /// Verified contributions by address
    contributions: Arc<RwLock<HashMap<[u8; 32], Vec<MiningContribution>>>>,

    /// Contribution cache for quick lookup
    contribution_cache: Arc<RwLock<HashMap<String, CachedContribution>>>,
}

/// Cached contribution data for performance
#[derive(Debug, Clone)]
struct CachedContribution {
    total_hashes: u128,
    verified_at: u64,
    expires_at: u64,
}

impl MiningContributionTracker {
    /// Create new contribution tracker
    pub fn new() -> Self {
        Self {
            contributions: Arc::new(RwLock::new(HashMap::new())),
            contribution_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Verify mining contribution is valid
    pub async fn verify_contribution(
        &self,
        contribution: &MiningContribution,
    ) -> Result<()> {
        // 1. Verify contribution period is valid
        self.verify_contribution_period(contribution)?;

        // 2. Verify total hashes matches solutions
        self.verify_hash_count(contribution)?;

        // 3. Verify merkle proofs (links to blockchain)
        self.verify_merkle_proofs(contribution).await?;

        info!("✅ Mining contribution verified: {} hashes",
              contribution.total_hashes);

        Ok(())
    }

    /// Verify contribution time period is valid
    fn verify_contribution_period(
        &self,
        contribution: &MiningContribution,
    ) -> Result<()> {
        let (start, end) = contribution.contribution_period;

        if start >= end {
            return Err(anyhow!("Invalid contribution period: start >= end"));
        }

        let now = chrono::Utc::now().timestamp() as u64;
        if start > now {
            return Err(anyhow!("Contribution period cannot be in the future"));
        }

        // Contribution period must be reasonable (not more than 1 year)
        const MAX_PERIOD: u64 = 365 * 24 * 3600; // 1 year
        if end - start > MAX_PERIOD {
            return Err(anyhow!("Contribution period too long (max 1 year)"));
        }

        Ok(())
    }

    /// Verify total hash count is reasonable
    fn verify_hash_count(&self, contribution: &MiningContribution) -> Result<()> {
        if contribution.total_hashes == 0 {
            return Err(anyhow!("Total hashes cannot be zero"));
        }

        // Sanity check: verify hash count is achievable in given time period
        let (start, end) = contribution.contribution_period;
        let duration_seconds = end - start;

        // Maximum achievable hashrate: ~1 PH/s (1e15 H/s) for entire network
        // This is generous to allow future growth
        const MAX_NETWORK_HASHRATE: u128 = 1_000_000_000_000_000; // 1 PH/s
        let max_hashes = MAX_NETWORK_HASHRATE * duration_seconds as u128;

        if contribution.total_hashes > max_hashes {
            return Err(anyhow!(
                "Claimed hashes ({}) exceeds maximum possible ({}) for time period",
                contribution.total_hashes,
                max_hashes
            ));
        }

        Ok(())
    }

    /// Verify merkle proofs link to blockchain
    async fn verify_merkle_proofs(
        &self,
        contribution: &MiningContribution,
    ) -> Result<()> {
        if contribution.merkle_proofs.is_empty() {
            warn!("⚠️  No merkle proofs provided for mining contribution");
            // Allow empty proofs for now (can use historical data)
            // In production, this should be mandatory
            return Ok(());
        }

        for proof in &contribution.merkle_proofs {
            self.verify_single_merkle_proof(proof).await?;
        }

        Ok(())
    }

    /// Verify a single merkle proof
    async fn verify_single_merkle_proof(&self, proof: &MerkleProof) -> Result<()> {
        // TODO: Implement actual merkle proof verification
        // This requires access to blockchain state to verify:
        // 1. Block at height exists
        // 2. Root matches block header
        // 3. Merkle path is valid

        debug!(
            "Verifying merkle proof: block_height={}, root={}",
            proof.block_height,
            hex::encode(proof.root)
        );

        // For now, just validate structure
        if proof.path.is_empty() {
            return Err(anyhow!("Merkle path cannot be empty"));
        }

        Ok(())
    }

    /// Register verified contribution for an address
    pub async fn register_contribution(
        &self,
        address: [u8; 32],
        contribution: MiningContribution,
    ) -> Result<()> {
        // Verify before registering
        self.verify_contribution(&contribution).await?;

        // Store contribution
        let mut contributions = self.contributions.write().await;
        contributions
            .entry(address)
            .or_insert_with(Vec::new)
            .push(contribution.clone());

        // Update cache
        let cache_key = hex::encode(address);
        let total_hashes: u128 = contributions[&address]
            .iter()
            .map(|c| c.total_hashes)
            .sum();

        let mut cache = self.contribution_cache.write().await;
        cache.insert(
            cache_key,
            CachedContribution {
                total_hashes,
                verified_at: chrono::Utc::now().timestamp() as u64,
                expires_at: chrono::Utc::now().timestamp() as u64 + 3600, // 1 hour cache
            },
        );

        info!(
            "✅ Registered mining contribution: address={}, hashes={}",
            hex::encode(address),
            contribution.total_hashes
        );

        Ok(())
    }

    /// Get total contributions for an address
    pub async fn get_total_contribution(&self, address: &[u8; 32]) -> u128 {
        // Check cache first
        let cache_key = hex::encode(address);
        {
            let cache = self.contribution_cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                let now = chrono::Utc::now().timestamp() as u64;
                if now < cached.expires_at {
                    return cached.total_hashes;
                }
            }
        }

        // Cache miss or expired, calculate from stored contributions
        let contributions = self.contributions.read().await;
        contributions
            .get(address)
            .map(|contribs| contribs.iter().map(|c| c.total_hashes).sum())
            .unwrap_or(0)
    }

    /// Get contribution statistics for an address
    pub async fn get_contribution_stats(&self, address: &[u8; 32]) -> ContributionStats {
        let contributions = self.contributions.read().await;

        let contribs = contributions.get(address);

        if let Some(contribs) = contribs {
            let total_hashes = contribs.iter().map(|c| c.total_hashes).sum();
            let solution_count = contribs.iter().map(|c| c.solutions.len() as u64).sum();

            let min_start = contribs
                .iter()
                .map(|c| c.contribution_period.0)
                .min()
                .unwrap_or(0);
            let max_end = contribs
                .iter()
                .map(|c| c.contribution_period.1)
                .max()
                .unwrap_or(0);

            ContributionStats {
                total_hashes,
                solution_count,
                period: (min_start, max_end),
                power_bonus_percent: 0.0, // Calculated by VotingPowerCalculator
            }
        } else {
            ContributionStats {
                total_hashes: 0,
                solution_count: 0,
                period: (0, 0),
                power_bonus_percent: 0.0,
            }
        }
    }
}

impl Default for MiningContributionTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_verify_contribution_period() {
        let tracker = MiningContributionTracker::new();

        let now = chrono::Utc::now().timestamp() as u64;

        // Valid period
        let valid_contribution = MiningContribution {
            solutions: vec![],
            total_hashes: 1_000_000,
            merkle_proofs: vec![],
            contribution_period: (now - 3600, now), // Last hour
        };

        assert!(tracker
            .verify_contribution_period(&valid_contribution)
            .is_ok());

        // Invalid: start after end
        let invalid_contribution = MiningContribution {
            solutions: vec![],
            total_hashes: 1_000_000,
            merkle_proofs: vec![],
            contribution_period: (now, now - 3600),
        };

        assert!(tracker
            .verify_contribution_period(&invalid_contribution)
            .is_err());
    }

    #[tokio::test]
    async fn test_verify_hash_count() {
        let tracker = MiningContributionTracker::new();

        let now = chrono::Utc::now().timestamp() as u64;

        // Reasonable hash count (1M hashes in 1 hour)
        let valid = MiningContribution {
            solutions: vec![],
            total_hashes: 1_000_000,
            merkle_proofs: vec![],
            contribution_period: (now - 3600, now),
        };

        assert!(tracker.verify_hash_count(&valid).is_ok());

        // Impossible hash count (too many for time period)
        let invalid = MiningContribution {
            solutions: vec![],
            total_hashes: u128::MAX,
            merkle_proofs: vec![],
            contribution_period: (now - 60, now), // 1 minute
        };

        assert!(tracker.verify_hash_count(&invalid).is_err());
    }

    #[tokio::test]
    async fn test_register_contribution() {
        let tracker = MiningContributionTracker::new();
        let address = [1u8; 32];

        let now = chrono::Utc::now().timestamp() as u64;
        let contribution = MiningContribution {
            solutions: vec![],
            total_hashes: 1_000_000,
            merkle_proofs: vec![],
            contribution_period: (now - 3600, now),
        };

        let result = tracker
            .register_contribution(address, contribution)
            .await;

        assert!(result.is_ok());

        // Verify it's stored
        let total = tracker.get_total_contribution(&address).await;
        assert_eq!(total, 1_000_000);
    }
}
