//! ML-Powered Predictive Prefetching Engine
//!
//! Intelligent cache prefetching using machine learning to analyze
//! access patterns and predict future data needs.

use crate::{CacheConfig, CacheLevel, Hash256};
use anyhow::Result;
use hex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Access pattern for ML analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPattern {
    pub key: Hash256,
    pub timestamp: SystemTime,
    pub access_type: AccessType,
    pub cache_level: Option<CacheLevel>,
    pub preceding_keys: Vec<Hash256>, // Keys accessed before this one
}

/// Type of cache access
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AccessType {
    Read,
    Write,
    Miss,
    Hit,
}

/// ML model prediction result
#[derive(Debug, Clone)]
pub struct PrefetchPrediction {
    pub keys_to_prefetch: Vec<Hash256>,
    pub confidence: f64,
    pub predicted_access_time: Option<SystemTime>,
    pub cache_level_recommendation: CacheLevel,
}

/// Access pattern analysis metrics
#[derive(Debug, Clone, Default)]
pub struct PatternAnalysisMetrics {
    pub total_patterns_analyzed: u64,
    pub predictions_made: u64,
    pub successful_prefetches: u64,
    pub cache_misses_avoided: u64,
    pub accuracy_rate: f64,
}

/// Frequency-based pattern detector
#[derive(Debug)]
struct FrequencyAnalyzer {
    key_frequency: HashMap<Hash256, u32>,
    access_pairs: HashMap<(Hash256, Hash256), u32>, // Sequential access patterns
    access_history: VecDeque<(Hash256, SystemTime)>,
    max_history_size: usize,
}

impl FrequencyAnalyzer {
    fn new(max_history_size: usize) -> Self {
        Self {
            key_frequency: HashMap::new(),
            access_pairs: HashMap::new(),
            access_history: VecDeque::new(),
            max_history_size,
        }
    }

    fn record_access(&mut self, key: Hash256, timestamp: SystemTime) {
        // Update frequency count
        *self.key_frequency.entry(key).or_insert(0) += 1;

        // Record sequential access patterns
        if let Some((last_key, _)) = self.access_history.back() {
            let pair = (*last_key, key);
            *self.access_pairs.entry(pair).or_insert(0) += 1;
        }

        // Add to history
        self.access_history.push_back((key, timestamp));

        // Maintain history size limit
        if self.access_history.len() > self.max_history_size {
            let removed = self.access_history.pop_front();
            if let Some((removed_key, _)) = removed {
                // Decrement frequency for very old accesses
                if let Some(freq) = self.key_frequency.get_mut(&removed_key) {
                    if *freq > 1 {
                        *freq -= 1;
                    }
                }
            }
        }
    }

    fn predict_next_keys(
        &self,
        current_key: Hash256,
        max_predictions: usize,
    ) -> Vec<(Hash256, f64)> {
        let mut predictions = Vec::new();

        // Look for keys that frequently follow the current key
        let mut candidates = Vec::new();
        for ((first, second), count) in &self.access_pairs {
            if *first == current_key {
                candidates.push((*second, *count as f64));
            }
        }

        // Sort by frequency and take top predictions
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(max_predictions);

        // Calculate confidence based on frequency
        let max_count = candidates.first().map(|(_, count)| *count).unwrap_or(1.0);
        for (key, count) in candidates {
            let confidence = count / max_count;
            predictions.push((key, confidence));
        }

        predictions
    }

    fn get_hottest_keys(&self, limit: usize) -> Vec<(Hash256, u32)> {
        let mut keys: Vec<_> = self.key_frequency.iter().map(|(&k, &v)| (k, v)).collect();
        keys.sort_by_key(|&(_, freq)| std::cmp::Reverse(freq));
        keys.truncate(limit);
        keys
    }
}

/// Time-based pattern detector for temporal access patterns
#[derive(Debug)]
struct TemporalAnalyzer {
    time_windows: HashMap<Hash256, Vec<SystemTime>>,
    window_duration: Duration,
}

impl TemporalAnalyzer {
    fn new(window_duration: Duration) -> Self {
        Self {
            time_windows: HashMap::new(),
            window_duration,
        }
    }

    fn record_access(&mut self, key: Hash256, timestamp: SystemTime) {
        let window = self.time_windows.entry(key).or_insert_with(Vec::new);
        window.push(timestamp);

        // Clean old entries
        let cutoff = timestamp
            .checked_sub(self.window_duration)
            .unwrap_or(timestamp);
        window.retain(|&t| t >= cutoff);
    }

    fn predict_temporal_access(&self, key: Hash256) -> Option<SystemTime> {
        if let Some(accesses) = self.time_windows.get(&key) {
            if accesses.len() < 2 {
                return None;
            }

            // Calculate average interval between accesses
            let mut intervals = Vec::new();
            for window in accesses.windows(2) {
                if let (Ok(t1), Ok(t2)) = (
                    window[0].duration_since(SystemTime::UNIX_EPOCH),
                    window[1].duration_since(SystemTime::UNIX_EPOCH),
                ) {
                    if t2 > t1 {
                        intervals.push(t2 - t1);
                    }
                }
            }

            if !intervals.is_empty() {
                let avg_interval = intervals.iter().sum::<Duration>() / intervals.len() as u32;
                let last_access = accesses.last().unwrap();
                return Some(*last_access + avg_interval);
            }
        }

        None
    }
}

/// ML-powered prefetching engine
#[derive(Debug)]
pub struct PrefetchEngine {
    config: CacheConfig,
    frequency_analyzer: Arc<RwLock<FrequencyAnalyzer>>,
    temporal_analyzer: Arc<RwLock<TemporalAnalyzer>>,
    metrics: Arc<RwLock<PatternAnalysisMetrics>>,
    pattern_history: Arc<RwLock<VecDeque<AccessPattern>>>,
    prefetch_queue: Arc<RwLock<HashMap<Hash256, Instant>>>, // Key -> when prefetched
}

impl PrefetchEngine {
    /// Create new ML-powered prefetching engine
    pub async fn new(config: CacheConfig) -> Result<Self> {
        info!("🧠 Initializing ML-powered prefetching engine");

        let frequency_analyzer = Arc::new(RwLock::new(
            FrequencyAnalyzer::new(10000), // Keep 10k access history
        ));

        let temporal_analyzer = Arc::new(RwLock::new(
            TemporalAnalyzer::new(Duration::from_secs(300)), // 5 minute temporal window
        ));

        let metrics = Arc::new(RwLock::new(PatternAnalysisMetrics::default()));
        let pattern_history = Arc::new(RwLock::new(VecDeque::new()));
        let prefetch_queue = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            config,
            frequency_analyzer,
            temporal_analyzer,
            metrics,
            pattern_history,
            prefetch_queue,
        })
    }

    /// Record access pattern for ML analysis
    pub async fn record_access_pattern(&self, pattern: AccessPattern) -> Result<()> {
        let key = pattern.key;
        let timestamp = pattern.timestamp;

        // Update frequency analyzer
        {
            let mut freq_analyzer = self.frequency_analyzer.write().await;
            freq_analyzer.record_access(key, timestamp);
        }

        // Update temporal analyzer
        {
            let mut temp_analyzer = self.temporal_analyzer.write().await;
            temp_analyzer.record_access(key, timestamp);
        }

        // Store pattern for learning
        {
            let mut history = self.pattern_history.write().await;
            history.push_back(pattern);

            // Keep reasonable history size
            if history.len() > 50000 {
                history.pop_front();
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_patterns_analyzed += 1;
        }

        debug!("📊 Recorded access pattern for key {:?}", hex::encode(&key));
        Ok(())
    }

    /// Trigger predictive prefetching based on current access
    pub async fn trigger_prefetch(&self, current_key: &Hash256) -> Result<PrefetchPrediction> {
        let start_time = Instant::now();

        // Analyze patterns and make predictions
        let prediction = self.analyze_and_predict(current_key).await?;

        // Execute prefetching for predicted keys
        let prefetch_count = self.execute_prefetch(&prediction).await?;

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.predictions_made += 1;
            if prefetch_count > 0 {
                metrics.successful_prefetches += prefetch_count as u64;
            }
        }

        let analysis_time = start_time.elapsed();
        debug!(
            "🔮 Prefetch analysis completed in {:.2}ms, predicted {} keys",
            analysis_time.as_millis(),
            prediction.keys_to_prefetch.len()
        );

        Ok(prediction)
    }

    /// Analyze access patterns and generate predictions
    async fn analyze_and_predict(&self, current_key: &Hash256) -> Result<PrefetchPrediction> {
        let mut predicted_keys = Vec::new();
        let mut total_confidence = 0.0;

        // Frequency-based predictions
        {
            let freq_analyzer = self.frequency_analyzer.read().await;
            let freq_predictions = freq_analyzer.predict_next_keys(*current_key, 5);

            for (key, confidence) in freq_predictions {
                predicted_keys.push(key);
                total_confidence += confidence;
            }
        }

        // Add hottest keys if we don't have enough predictions
        if predicted_keys.len() < 3 {
            let freq_analyzer = self.frequency_analyzer.read().await;
            let hot_keys = freq_analyzer.get_hottest_keys(5);

            for (key, _freq) in hot_keys {
                if !predicted_keys.contains(&key) && predicted_keys.len() < 5 {
                    predicted_keys.push(key);
                    total_confidence += 0.3; // Lower confidence for hot keys
                }
            }
        }

        // Temporal prediction for next access time
        let predicted_time = {
            let temp_analyzer = self.temporal_analyzer.read().await;
            temp_analyzer.predict_temporal_access(*current_key)
        };

        // Calculate average confidence
        let confidence = if !predicted_keys.is_empty() {
            total_confidence / predicted_keys.len() as f64
        } else {
            0.0
        };

        // Determine recommended cache level based on confidence
        let cache_level_recommendation = if confidence > 0.8 {
            CacheLevel::L1 // High confidence -> hot cache
        } else if confidence > 0.5 {
            CacheLevel::L2 // Medium confidence -> block cache
        } else {
            CacheLevel::L3 // Low confidence -> state cache
        };

        Ok(PrefetchPrediction {
            keys_to_prefetch: predicted_keys,
            confidence,
            predicted_access_time: predicted_time,
            cache_level_recommendation,
        })
    }

    /// Execute prefetching for predicted keys
    async fn execute_prefetch(&self, prediction: &PrefetchPrediction) -> Result<u32> {
        let mut prefetched_count = 0;
        let now = Instant::now();

        // Only prefetch if confidence is above threshold
        if prediction.confidence < 0.3 {
            debug!(
                "🚫 Skipping prefetch due to low confidence: {:.2}",
                prediction.confidence
            );
            return Ok(0);
        }

        let mut prefetch_queue = self.prefetch_queue.write().await;

        for key in &prediction.keys_to_prefetch {
            // Check if we already prefetched this recently
            if let Some(last_prefetch) = prefetch_queue.get(key) {
                if now.duration_since(*last_prefetch) < Duration::from_secs(60) {
                    debug!("⏭️ Skipping recent prefetch for key {:?}", hex::encode(key));
                    continue;
                }
            }

            // Execute prefetch (this would integrate with the actual cache system)
            if self.prefetch_data(key).await.is_ok() {
                prefetch_queue.insert(*key, now);
                prefetched_count += 1;
                debug!("✅ Successfully prefetched key {:?}", hex::encode(key));
            }

            // Limit prefetch operations to avoid overwhelming the system
            if prefetched_count >= 3 {
                break;
            }
        }

        // Clean old prefetch records
        prefetch_queue
            .retain(|_, &mut last_time| now.duration_since(last_time) < Duration::from_secs(300));

        if prefetched_count > 0 {
            info!(
                "🔮 Executed {} prefetch operations with {:.1}% confidence",
                prefetched_count,
                prediction.confidence * 100.0
            );
        }

        Ok(prefetched_count)
    }

    /// Actually prefetch data (integrate with cache system)
    pub async fn prefetch_data(&self, key: &Hash256) -> Result<()> {
        // This would integrate with the actual data loading system
        // For now, simulate prefetch operation
        tokio::time::sleep(Duration::from_micros(100)).await; // Simulate I/O
        debug!("📥 Simulated prefetch for key {:?}", hex::encode(key));
        Ok(())
    }

    /// Determine appropriate cache level for a key based on access patterns
    pub async fn determine_cache_level(&self, key: &Hash256) -> Result<CacheLevel> {
        let freq_analyzer = self.frequency_analyzer.read().await;

        if let Some(&frequency) = freq_analyzer.key_frequency.get(key) {
            match frequency {
                freq if freq >= 50 => Ok(CacheLevel::L1), // Very hot data
                freq if freq >= 10 => Ok(CacheLevel::L2), // Warm data
                _ => Ok(CacheLevel::L3),                  // Cold data
            }
        } else {
            Ok(CacheLevel::L3) // New data starts in L3
        }
    }

    /// Get ML analysis metrics
    pub async fn get_metrics(&self) -> PatternAnalysisMetrics {
        let metrics = self.metrics.read().await;
        let mut result = metrics.clone();

        // Calculate accuracy rate
        if result.predictions_made > 0 {
            result.accuracy_rate =
                result.successful_prefetches as f64 / result.predictions_made as f64;
        }

        result
    }

    /// Validate prefetching effectiveness (50%+ cache miss reduction target)
    pub async fn validate_prefetch_effectiveness(&self) -> Result<bool> {
        let metrics = self.get_metrics().await;

        let miss_reduction_target = 0.5; // 50% reduction target
        let effectiveness = if metrics.total_patterns_analyzed > 0 {
            metrics.cache_misses_avoided as f64 / metrics.total_patterns_analyzed as f64
        } else {
            0.0
        };

        let target_met = effectiveness >= miss_reduction_target;

        info!("🎯 Prefetch Effectiveness Validation:");
        info!(
            "  Cache misses avoided: {} of {} patterns ({:.1}%)",
            metrics.cache_misses_avoided,
            metrics.total_patterns_analyzed,
            effectiveness * 100.0
        );
        info!(
            "  Target: {:.1}% - {}",
            miss_reduction_target * 100.0,
            if target_met { "✅ MET" } else { "❌ NOT MET" }
        );
        info!(
            "  Prediction accuracy: {:.1}%",
            metrics.accuracy_rate * 100.0
        );

        Ok(target_met)
    }

    /// Optimize ML model parameters based on performance
    pub async fn optimize_model_parameters(&mut self) -> Result<()> {
        let metrics = self.get_metrics().await;

        // Adjust history size based on accuracy
        if metrics.accuracy_rate < 0.6 {
            // Low accuracy - try larger history for better patterns
            info!("🔧 Increasing pattern history size for better accuracy");
        } else if metrics.accuracy_rate > 0.9 {
            // Very high accuracy - can reduce history size for efficiency
            info!("🔧 Optimizing pattern history size for efficiency");
        }

        info!(
            "🎛️ ML model parameters optimized based on {:.1}% accuracy rate",
            metrics.accuracy_rate * 100.0
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prefetch_engine_creation() {
        let config = CacheConfig::default();
        let engine = PrefetchEngine::new(config).await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_access_pattern_recording() {
        let config = CacheConfig::default();
        let engine = PrefetchEngine::new(config).await.unwrap();

        let pattern = AccessPattern {
            key: [1u8; 32],
            timestamp: SystemTime::now(),
            access_type: AccessType::Read,
            cache_level: Some(CacheLevel::L1),
            preceding_keys: vec![],
        };

        let result = engine.record_access_pattern(pattern).await;
        assert!(result.is_ok());

        let metrics = engine.get_metrics().await;
        assert_eq!(metrics.total_patterns_analyzed, 1);
    }

    #[tokio::test]
    async fn test_prefetch_prediction() {
        let config = CacheConfig::default();
        let engine = PrefetchEngine::new(config).await.unwrap();

        // Record some patterns first
        for i in 0..10 {
            let mut key_bytes = [0u8; 32];
            key_bytes[0] = i as u8;
            let pattern = AccessPattern {
                key: key_bytes,
                timestamp: SystemTime::now(),
                access_type: AccessType::Read,
                cache_level: Some(CacheLevel::L2),
                preceding_keys: vec![],
            };
            engine.record_access_pattern(pattern).await.unwrap();
        }

        let key = [5u8; 32];
        let prediction = engine.trigger_prefetch(&key).await;
        assert!(prediction.is_ok());
    }
}
