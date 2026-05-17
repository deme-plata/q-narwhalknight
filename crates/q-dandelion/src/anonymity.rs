//! Anonymity Metrics and Analysis for Dandelion++ Protocol

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::DandelionPhase;

/// Anonymity metrics for Dandelion++ protocol
#[derive(Debug)]
pub struct AnonymityMetrics {
    /// Total messages processed
    pub total_messages: AtomicU64,
    /// Messages in stem phase
    pub stem_messages: AtomicU64,
    /// Messages in fluff phase
    pub fluff_messages: AtomicU64,
    /// Hop count distribution
    hop_distribution: Arc<RwLock<HashMap<u8, u64>>>,
    /// Average delay in milliseconds
    average_delay_ms: Arc<RwLock<f64>>,
    /// Computed anonymity score
    anonymity_score: Arc<RwLock<f64>>,
}

impl AnonymityMetrics {
    pub fn new() -> Self {
        Self {
            total_messages: AtomicU64::new(0),
            stem_messages: AtomicU64::new(0),
            fluff_messages: AtomicU64::new(0),
            hop_distribution: Arc::new(RwLock::new(HashMap::new())),
            average_delay_ms: Arc::new(RwLock::new(0.0)),
            anonymity_score: Arc::new(RwLock::new(0.0)),
        }
    }

    /// Record a message propagation event
    pub async fn record_message_propagation(&self, phase: DandelionPhase, hop_count: u8) {
        self.total_messages.fetch_add(1, Ordering::Relaxed);

        match phase {
            DandelionPhase::Stem => {
                self.stem_messages.fetch_add(1, Ordering::Relaxed);
            }
            DandelionPhase::Fluff => {
                self.fluff_messages.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Update hop distribution
        {
            let mut dist = self.hop_distribution.write().await;
            *dist.entry(hop_count).or_insert(0) += 1;
        }

        debug!(
            "Recorded {:?} message with {} hops",
            phase, hop_count
        );
    }

    /// Record a message with delay information
    pub async fn record_message(&self, is_stem: bool, delay_ms: f64) -> Result<()> {
        let total = self.total_messages.fetch_add(1, Ordering::Relaxed) + 1;

        if is_stem {
            self.stem_messages.fetch_add(1, Ordering::Relaxed);
        } else {
            self.fluff_messages.fetch_add(1, Ordering::Relaxed);
        }

        // Update average delay (exponential moving average)
        {
            let mut avg_delay = self.average_delay_ms.write().await;
            *avg_delay = *avg_delay * 0.9 + delay_ms * 0.1;
        }

        debug!("Recorded message: stem={}, delay={}ms", is_stem, delay_ms);
        Ok(())
    }

    /// Calculate and update anonymity score
    pub async fn calculate_anonymity_score(&self) -> f64 {
        let total = self.total_messages.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }

        let stem = self.stem_messages.load(Ordering::Relaxed);
        let hop_dist = self.hop_distribution.read().await;

        // Calculate entropy of hop distribution
        let hop_entropy = Self::calculate_entropy(&hop_dist);

        // Factor in stem ratio (higher stem ratio = better anonymity)
        let stem_ratio = stem as f64 / total as f64;

        // Combine factors into score (0.0 - 1.0)
        let score = (stem_ratio * 0.5 + hop_entropy * 0.5).min(1.0);

        // Store the score
        {
            let mut anonymity_score = self.anonymity_score.write().await;
            *anonymity_score = score;
        }

        score
    }

    /// Calculate entropy of a distribution
    fn calculate_entropy(distribution: &HashMap<u8, u64>) -> f64 {
        let total: u64 = distribution.values().sum();
        if total == 0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for count in distribution.values() {
            if *count > 0 {
                let p = (*count as f64) / (total as f64);
                entropy -= p * p.log2();
            }
        }

        // Normalize to 0-1 range (assuming max 10 hops = ~3.32 bits max entropy)
        (entropy / 3.32).min(1.0)
    }

    /// Get current metrics snapshot
    pub async fn get_current_metrics(&self) -> AnonymityMetricsSnapshot {
        let hop_distribution = self.hop_distribution.read().await.clone();
        let average_delay_ms = *self.average_delay_ms.read().await;
        let anonymity_score = *self.anonymity_score.read().await;

        AnonymityMetricsSnapshot {
            total_messages: self.total_messages.load(Ordering::Relaxed),
            stem_messages: self.stem_messages.load(Ordering::Relaxed),
            fluff_messages: self.fluff_messages.load(Ordering::Relaxed),
            hop_distribution,
            average_delay_ms,
            anonymity_score,
        }
    }
}

impl Default for AnonymityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of anonymity metrics at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymityMetricsSnapshot {
    pub total_messages: u64,
    pub stem_messages: u64,
    pub fluff_messages: u64,
    pub hop_distribution: HashMap<u8, u64>,
    pub average_delay_ms: f64,
    pub anonymity_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_anonymity_metrics_recording() {
        let metrics = AnonymityMetrics::new();

        metrics.record_message_propagation(DandelionPhase::Stem, 2).await;
        metrics.record_message_propagation(DandelionPhase::Stem, 3).await;
        metrics.record_message_propagation(DandelionPhase::Fluff, 1).await;

        let snapshot = metrics.get_current_metrics().await;
        assert_eq!(snapshot.total_messages, 3);
        assert_eq!(snapshot.stem_messages, 2);
        assert_eq!(snapshot.fluff_messages, 1);
    }

    #[tokio::test]
    async fn test_hop_distribution() {
        let metrics = AnonymityMetrics::new();

        metrics.record_message_propagation(DandelionPhase::Stem, 2).await;
        metrics.record_message_propagation(DandelionPhase::Stem, 2).await;
        metrics.record_message_propagation(DandelionPhase::Stem, 3).await;

        let snapshot = metrics.get_current_metrics().await;
        assert_eq!(snapshot.hop_distribution.get(&2), Some(&2));
        assert_eq!(snapshot.hop_distribution.get(&3), Some(&1));
    }

    #[tokio::test]
    async fn test_anonymity_score_calculation() {
        let metrics = AnonymityMetrics::new();

        // All stem messages = high anonymity
        for _ in 0..10 {
            metrics.record_message_propagation(DandelionPhase::Stem, 3).await;
        }

        let score = metrics.calculate_anonymity_score().await;
        assert!(score > 0.0); // Should have positive score
    }

    #[test]
    fn test_entropy_calculation() {
        let mut distribution = HashMap::new();
        distribution.insert(1, 10);
        distribution.insert(2, 10);
        distribution.insert(3, 10);

        let entropy = AnonymityMetrics::calculate_entropy(&distribution);
        assert!(entropy > 0.0);
        assert!(entropy <= 1.0);
    }
}
