// Network Hashrate Tracking Infrastructure
// Monitors mining activity and provides smoothed hashrate estimates for adaptive security

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Mining solution submission (from miners)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningSolution {
    pub wallet_address: String,
    pub nonce: u64,
    pub hash: Vec<u8>,
    pub timestamp: u64,
    pub difficulty: f64,
}

/// Hashrate snapshot for time-series tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkHashrateSnapshot {
    pub timestamp: u64,
    pub hashrate: f64,        // Hashes per second
    pub active_miners: usize, // Number of active miners
    pub block_height: u64,    // Current blockchain height
    pub difficulty: f64,      // Current mining difficulty
    pub solutions_5min: u64,  // Solutions in last 5 minutes
}

/// Network-wide hashrate tracker
pub struct NetworkHashrateTracker {
    /// Recent hashrate snapshots (24-hour window)
    snapshots: Arc<RwLock<VecDeque<NetworkHashrateSnapshot>>>,

    /// Recent mining solutions (for rate calculation)
    recent_solutions: Arc<RwLock<VecDeque<MiningSolution>>>,

    /// Snapshot window duration (24 hours)
    window_duration: Duration,

    /// Solution tracking window (5 minutes)
    solution_window: Duration,

    /// Current difficulty target
    current_difficulty: Arc<RwLock<f64>>,

    /// Active miner addresses
    active_miners: Arc<RwLock<std::collections::HashSet<String>>>,
}

impl NetworkHashrateTracker {
    pub fn new() -> Self {
        Self {
            snapshots: Arc::new(RwLock::new(VecDeque::new())),
            recent_solutions: Arc::new(RwLock::new(VecDeque::new())),
            window_duration: Duration::from_secs(86400), // 24 hours
            solution_window: Duration::from_secs(300),   // 5 minutes
            current_difficulty: Arc::new(RwLock::new(1.0)),
            active_miners: Arc::new(RwLock::new(std::collections::HashSet::new())),
        }
    }

    /// Record a mining solution submission
    pub async fn record_solution(&self, solution: MiningSolution) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Add to recent solutions
        {
            let mut solutions = self.recent_solutions.write().await;
            solutions.push_back(solution.clone());

            // Remove solutions older than 5 minutes
            let cutoff = now.saturating_sub(self.solution_window.as_secs());
            while let Some(front) = solutions.front() {
                if front.timestamp < cutoff {
                    solutions.pop_front();
                } else {
                    break;
                }
            }
        }

        // Update active miners
        {
            let mut miners = self.active_miners.write().await;
            miners.insert(solution.wallet_address.clone());
        }

        debug!(
            "Recorded mining solution from {} at height {}",
            solution.wallet_address, now
        );

        Ok(())
    }

    /// Compute current network hashrate based on recent solutions
    pub async fn compute_current_hashrate(&self) -> Result<f64> {
        let solutions = self.recent_solutions.read().await;
        let difficulty = *self.current_difficulty.read().await;

        if solutions.is_empty() {
            return Ok(1_000_000_000.0); // 1 GH/s baseline
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Count solutions in last 5 minutes
        let recent_count = solutions
            .iter()
            .filter(|s| now - s.timestamp <= 300)
            .count() as f64;

        // Estimate hashrate: (solutions × difficulty × 2^32) / time_window
        let time_window = 300.0; // 5 minutes in seconds
        let hash_attempts_per_solution = difficulty * (u32::MAX as f64);
        let estimated_hashrate = (recent_count * hash_attempts_per_solution) / time_window;

        Ok(estimated_hashrate)
    }

    /// Take a snapshot of current network hashrate
    pub async fn take_snapshot(&self, block_height: u64) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let hashrate = self.compute_current_hashrate().await?;
        let solutions_5min = self.recent_solutions.read().await.len() as u64;
        let active_miners = self.active_miners.read().await.len();
        let difficulty = *self.current_difficulty.read().await;

        let snapshot = NetworkHashrateSnapshot {
            timestamp: now,
            hashrate,
            active_miners,
            block_height,
            difficulty,
            solutions_5min,
        };

        // Add to snapshots
        {
            let mut snapshots = self.snapshots.write().await;
            snapshots.push_back(snapshot.clone());

            // Remove snapshots older than 24 hours
            let cutoff = now.saturating_sub(self.window_duration.as_secs());
            while let Some(front) = snapshots.front() {
                if front.timestamp < cutoff {
                    snapshots.pop_front();
                } else {
                    break;
                }
            }
        }

        info!(
            "Hashrate snapshot: {:.2} MH/s ({} miners, {} solutions/5min)",
            hashrate / 1_000_000.0,
            active_miners,
            solutions_5min
        );

        Ok(())
    }

    /// Compute 24-hour smoothed hashrate (moving average)
    pub async fn compute_smoothed_hashrate(&self) -> Result<f64> {
        let snapshots = self.snapshots.read().await;

        if snapshots.is_empty() {
            return Ok(1_000_000_000.0); // 1 GH/s baseline
        }

        // Compute weighted moving average (more recent snapshots have higher weight)
        let total_weight: f64 = snapshots.len() as f64;
        let weighted_sum: f64 = snapshots
            .iter()
            .enumerate()
            .map(|(i, snapshot)| {
                let weight = (i + 1) as f64 / total_weight; // Linear weighting
                snapshot.hashrate * weight
            })
            .sum();

        let weighted_avg = weighted_sum / (total_weight * (total_weight + 1.0) / 2.0);

        Ok(weighted_avg)
    }

    /// Update mining difficulty
    pub async fn update_difficulty(&self, new_difficulty: f64) -> Result<()> {
        let mut difficulty = self.current_difficulty.write().await;
        *difficulty = new_difficulty;

        info!("Mining difficulty updated to {:.6}", new_difficulty);
        Ok(())
    }

    /// Get active miner count
    pub async fn get_active_miner_count(&self) -> usize {
        self.active_miners.read().await.len()
    }

    /// Clean up inactive miners (not seen in 1 hour)
    pub async fn cleanup_inactive_miners(&self) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let cutoff = now.saturating_sub(3600); // 1 hour

        let recent_miners: std::collections::HashSet<String> = {
            let solutions = self.recent_solutions.read().await;
            solutions
                .iter()
                .filter(|s| s.timestamp >= cutoff)
                .map(|s| s.wallet_address.clone())
                .collect()
        };

        let mut miners = self.active_miners.write().await;
        *miners = recent_miners;

        Ok(())
    }

    /// Get hashrate statistics for API
    pub async fn get_statistics(&self) -> HashrateStatistics {
        let current = self.compute_current_hashrate().await.unwrap_or(0.0);
        let smoothed = self.compute_smoothed_hashrate().await.unwrap_or(0.0);
        let active_miners = self.get_active_miner_count().await;
        let recent_solutions = self.recent_solutions.read().await.len();
        let difficulty = *self.current_difficulty.read().await;

        HashrateStatistics {
            current_hashrate: current,
            smoothed_hashrate_24h: smoothed,
            active_miners,
            solutions_5min: recent_solutions,
            difficulty,
        }
    }

    /// Detect potential hashrate manipulation
    pub async fn detect_manipulation(&self) -> bool {
        let snapshots = self.snapshots.read().await;

        if snapshots.len() < 12 {
            return false; // Not enough data
        }

        // Check for sudden spikes (>100% change in <1 hour)
        let recent: Vec<_> = snapshots.iter().rev().take(12).collect(); // Last hour

        if recent.len() < 2 {
            return false;
        }

        let first = recent.last().unwrap().hashrate;
        let last = recent.first().unwrap().hashrate;

        if first < 1_000_000.0 {
            return false; // Too low to analyze
        }

        let change = (last - first).abs() / first;

        if change > 1.0 {
            // >100% change
            warn!(
                "🚨 Potential hashrate manipulation detected: {:.1}% change in 1 hour",
                change * 100.0
            );
            return true;
        }

        false
    }
}

/// Hashrate statistics for API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashrateStatistics {
    pub current_hashrate: f64,
    pub smoothed_hashrate_24h: f64,
    pub active_miners: usize,
    pub solutions_5min: usize,
    pub difficulty: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hashrate_tracker_creation() {
        let tracker = NetworkHashrateTracker::new();
        let stats = tracker.get_statistics().await;

        assert_eq!(stats.active_miners, 0);
        assert_eq!(stats.solutions_5min, 0);
    }

    #[tokio::test]
    async fn test_record_solution() {
        let tracker = NetworkHashrateTracker::new();

        let solution = MiningSolution {
            wallet_address: "test_wallet".to_string(),
            nonce: 12345,
            hash: vec![0u8; 32],
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            difficulty: 1.0,
        };

        tracker.record_solution(solution).await.unwrap();

        let stats = tracker.get_statistics().await;
        assert_eq!(stats.active_miners, 1);
        assert_eq!(stats.solutions_5min, 1);
    }

    #[tokio::test]
    async fn test_smoothed_hashrate() {
        let tracker = NetworkHashrateTracker::new();

        // Simulate 10 snapshots
        for i in 0..10 {
            tracker.take_snapshot(i).await.unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        let smoothed = tracker.compute_smoothed_hashrate().await.unwrap();
        assert!(smoothed > 0.0);
    }

    #[tokio::test]
    async fn test_difficulty_update() {
        let tracker = NetworkHashrateTracker::new();

        tracker.update_difficulty(2.5).await.unwrap();

        let stats = tracker.get_statistics().await;
        assert_eq!(stats.difficulty, 2.5);
    }
}
