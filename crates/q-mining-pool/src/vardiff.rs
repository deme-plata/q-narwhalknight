//! Variable difficulty controller for mining pool

use std::collections::VecDeque;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

use crate::config::VardiffConfig;

/// Variable difficulty controller
///
/// Adjusts share difficulty based on worker performance to maintain
/// a target share submission rate.
#[derive(Debug, Clone)]
pub struct VardiffController {
    /// Configuration
    config: VardiffConfig,

    /// Current difficulty
    current_difficulty: f64,

    /// Share timestamps for rate calculation
    share_times: VecDeque<Instant>,

    /// Last retarget time
    last_retarget: Instant,

    /// Total shares since last retarget
    shares_since_retarget: u64,
}

impl VardiffController {
    /// Create new vardiff controller
    pub fn new(config: VardiffConfig) -> Self {
        Self {
            current_difficulty: config.initial_difficulty,
            config,
            share_times: VecDeque::with_capacity(100),
            last_retarget: Instant::now(),
            shares_since_retarget: 0,
        }
    }

    /// Get current difficulty
    pub fn difficulty(&self) -> f64 {
        self.current_difficulty
    }

    /// Record a share accepted
    pub fn on_share_accepted(&mut self) {
        let now = Instant::now();
        self.share_times.push_back(now);
        self.shares_since_retarget += 1;

        // Keep only last 100 shares for rate calculation
        while self.share_times.len() > 100 {
            self.share_times.pop_front();
        }
    }

    /// Check if it's time to retarget
    pub fn should_retarget(&self) -> bool {
        self.last_retarget.elapsed() >= Duration::from_secs_f64(self.config.retarget_interval_seconds)
    }

    /// Calculate average time between shares
    pub fn average_share_time(&self) -> f64 {
        if self.share_times.len() < 2 {
            return self.config.target_time_seconds;
        }

        let first = *self.share_times.front().unwrap();
        let last = *self.share_times.back().unwrap();
        let elapsed = last.duration_since(first).as_secs_f64();

        elapsed / (self.share_times.len() - 1) as f64
    }

    /// Calculate new difficulty if retarget needed
    pub fn calculate_new_difficulty(&mut self) -> Option<f64> {
        if !self.should_retarget() {
            return None;
        }

        let elapsed = self.last_retarget.elapsed().as_secs_f64();
        let shares = self.shares_since_retarget;

        // Reset for next interval
        self.last_retarget = Instant::now();
        self.shares_since_retarget = 0;

        if shares == 0 {
            // No shares - decrease difficulty significantly
            let new_diff = (self.current_difficulty * 0.5)
                .max(self.config.min_difficulty);

            tracing::debug!(
                old = self.current_difficulty,
                new = new_diff,
                reason = "no shares",
                "Vardiff adjustment"
            );

            self.current_difficulty = new_diff;
            return Some(new_diff);
        }

        // Calculate actual share rate
        let actual_time_per_share = elapsed / shares as f64;
        let target_time = self.config.target_time_seconds;

        // Calculate ratio (how far off we are from target)
        let ratio = actual_time_per_share / target_time;

        // Only adjust if outside variance threshold
        if ratio > 1.0 + self.config.variance_percent {
            // Shares coming too slow, decrease difficulty
            let adjustment = ratio.min(2.0); // Cap adjustment at 2x
            let new_diff = (self.current_difficulty / adjustment)
                .max(self.config.min_difficulty);

            tracing::debug!(
                old = self.current_difficulty,
                new = new_diff,
                ratio = ratio,
                actual_time = actual_time_per_share,
                target_time = target_time,
                "Vardiff decrease"
            );

            self.current_difficulty = new_diff;
            Some(new_diff)
        } else if ratio < 1.0 - self.config.variance_percent {
            // Shares coming too fast, increase difficulty
            let adjustment = (1.0 / ratio).min(2.0); // Cap adjustment at 2x
            let new_diff = (self.current_difficulty * adjustment)
                .min(self.config.max_difficulty);

            tracing::debug!(
                old = self.current_difficulty,
                new = new_diff,
                ratio = ratio,
                actual_time = actual_time_per_share,
                target_time = target_time,
                "Vardiff increase"
            );

            self.current_difficulty = new_diff;
            Some(new_diff)
        } else {
            // Within acceptable range, no change
            None
        }
    }

    /// Force set difficulty (for manual override)
    pub fn set_difficulty(&mut self, difficulty: f64) {
        self.current_difficulty = difficulty
            .max(self.config.min_difficulty)
            .min(self.config.max_difficulty);
    }

    /// Get stats for monitoring
    pub fn stats(&self) -> VardiffStats {
        let retarget_interval = Duration::from_secs_f64(self.config.retarget_interval_seconds);
        VardiffStats {
            current_difficulty: self.current_difficulty,
            average_share_time: self.average_share_time(),
            target_share_time: self.config.target_time_seconds,
            shares_since_retarget: self.shares_since_retarget,
            time_until_retarget: retarget_interval.saturating_sub(self.last_retarget.elapsed()),
        }
    }
}

/// Vardiff statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VardiffStats {
    pub current_difficulty: f64,
    pub average_share_time: f64,
    pub target_share_time: f64,
    pub shares_since_retarget: u64,
    #[serde(with = "duration_serde")]
    pub time_until_retarget: Duration,
}

mod duration_serde {
    use serde::{self, Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(duration.as_secs_f64())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(deserializer)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> VardiffConfig {
        VardiffConfig {
            enabled: true,
            initial_difficulty: 1.0,
            target_time_seconds: 10.0,
            variance_percent: 0.25,
            min_difficulty: 0.001,
            max_difficulty: 1000.0,
            retarget_interval_seconds: 60.0,
        }
    }

    #[test]
    fn test_initial_difficulty() {
        let controller = VardiffController::new(test_config());
        assert_eq!(controller.difficulty(), 1.0);
    }

    #[test]
    fn test_share_recording() {
        let mut controller = VardiffController::new(test_config());

        for _ in 0..10 {
            controller.on_share_accepted();
        }

        assert_eq!(controller.shares_since_retarget, 10);
        assert_eq!(controller.share_times.len(), 10);
    }

    #[test]
    fn test_difficulty_bounds() {
        let mut controller = VardiffController::new(test_config());

        controller.set_difficulty(0.0001);
        assert_eq!(controller.difficulty(), 0.001); // min

        controller.set_difficulty(10000.0);
        assert_eq!(controller.difficulty(), 1000.0); // max
    }
}
