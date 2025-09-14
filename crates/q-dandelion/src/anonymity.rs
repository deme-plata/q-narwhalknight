//! Anonymity Metrics and Analysis

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Anonymity metrics for Dandelion++ protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymityMetrics {
    pub total_messages: u64,
    pub stem_messages: u64,
    pub fluff_messages: u64,
    pub average_delay: f64,
    pub anonymity_score: f64,
}

impl AnonymityMetrics {
    pub fn new() -> Self {
        Self {
            total_messages: 0,
            stem_messages: 0,
            fluff_messages: 0,
            average_delay: 0.0,
            anonymity_score: 0.0,
        }
    }

    pub async fn record_message(&mut self, is_stem: bool, delay: f64) -> Result<()> {
        self.total_messages += 1;
        if is_stem {
            self.stem_messages += 1;
        } else {
            self.fluff_messages += 1;
        }

        // Update average delay
        self.average_delay = (self.average_delay * (self.total_messages - 1) as f64 + delay)
            / self.total_messages as f64;

        debug!("Recorded message: stem={}, delay={}", is_stem, delay);
        Ok(())
    }

    pub fn calculate_anonymity_score(&mut self) -> f64 {
        // Simple anonymity score calculation
        if self.total_messages == 0 {
            return 0.0;
        }

        let stem_ratio = self.stem_messages as f64 / self.total_messages as f64;
        self.anonymity_score = stem_ratio * 100.0;
        self.anonymity_score
    }

    pub async fn get_current_metrics(&self) -> AnonymityMetrics {
        self.clone()
    }
}
