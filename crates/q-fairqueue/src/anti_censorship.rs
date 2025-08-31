/// Anti-censorship mechanisms for quantum fair queueing

use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use tracing::{debug, warn, info};

use q_types::{NodeId, TransactionId};

/// Censorship resistance levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CensorshipResistance {
    /// No special protection
    None,
    
    /// Basic detection and logging
    Detection,
    
    /// Active countermeasures
    Active,
    
    /// Quantum-enhanced protection
    QuantumEnhanced,
}

/// Anti-censorship manager
pub struct AntiCensorshipManager {
    /// Resistance level
    resistance_level: CensorshipResistance,
    
    /// Suspected censoring nodes
    suspected_censors: HashSet<NodeId>,
    
    /// Transaction rejection patterns
    rejection_patterns: HashMap<NodeId, RejectionPattern>,
    
    /// Censorship detection metrics
    detection_metrics: CensorshipMetrics,
    
    /// Countermeasure strategies
    countermeasures: Vec<CountermeasureStrategy>,
}

/// Rejection pattern tracking
#[derive(Debug, Clone)]
struct RejectionPattern {
    total_rejections: u64,
    recent_rejections: Vec<Instant>,
    rejection_rate: f64,
    suspicious_patterns: Vec<SuspiciousPattern>,
}

/// Suspicious censorship patterns
#[derive(Debug, Clone)]
enum SuspiciousPattern {
    /// High rejection rate
    HighRejectionRate { rate: f64, window: Duration },
    
    /// Selective targeting of specific nodes
    SelectiveTargeting { target_nodes: HashSet<NodeId>, bias: f64 },
    
    /// Temporal patterns (censoring at specific times)
    TemporalPattern { time_windows: Vec<(u32, u32)> }, // (hour_start, hour_end)
    
    /// Content-based filtering
    ContentFiltering { pattern_hash: Vec<u8> },
}

/// Censorship detection metrics
#[derive(Debug, Clone, Default)]
struct CensorshipMetrics {
    total_censorship_events: u64,
    detected_censors: u64,
    false_positives: u64,
    countermeasures_applied: u64,
    success_rate: f64,
}

/// Countermeasure strategies
#[derive(Debug, Clone)]
enum CountermeasureStrategy {
    /// Route around suspected censors
    RouteAround { avoid_nodes: HashSet<NodeId> },
    
    /// Disguise transactions
    TransactionDisguise { obfuscation_level: u8 },
    
    /// Flood with decoy transactions
    DecoyFlooding { decoy_rate: u64 },
    
    /// Use quantum randomness to evade detection
    QuantumEvasion { randomization_strength: f64 },
    
    /// Reputation penalties for censors
    ReputationPenalty { penalty_factor: f64 },
}

impl AntiCensorshipManager {
    /// Create new anti-censorship manager
    pub fn new(resistance_enabled: bool) -> Result<Self> {
        let resistance_level = if resistance_enabled {
            CensorshipResistance::Active
        } else {
            CensorshipResistance::Detection
        };

        Ok(Self {
            resistance_level,
            suspected_censors: HashSet::new(),
            rejection_patterns: HashMap::new(),
            detection_metrics: CensorshipMetrics::default(),
            countermeasures: Vec::new(),
        })
    }

    /// Check if transaction/node combination shows censorship
    pub async fn is_censorship_attempt(&mut self, tx_id: &TransactionId, from_node: NodeId) -> Result<bool> {
        if matches!(self.resistance_level, CensorshipResistance::None) {
            return Ok(false);
        }

        // Check for suspicious patterns
        let is_suspicious = self.analyze_censorship_patterns(tx_id, from_node).await?;
        
        if is_suspicious {
            warn!("Potential censorship detected: tx={}, node={}", 
                  hex::encode(tx_id), hex::encode(&from_node));
            
            self.detection_metrics.total_censorship_events += 1;
            
            // Add to suspected censors if pattern is strong
            if self.is_strong_censorship_evidence(from_node).await? {
                self.suspected_censors.insert(from_node);
                self.detection_metrics.detected_censors += 1;
            }
        }

        Ok(is_suspicious)
    }

    /// Apply countermeasures against censorship
    pub async fn apply_countermeasures(&mut self, tx_id: &TransactionId, target_node: NodeId) -> Result<()> {
        if !matches!(self.resistance_level, CensorshipResistance::Active | CensorshipResistance::QuantumEnhanced) {
            return Ok(());
        }

        info!("Applying anti-censorship countermeasures for transaction {}", hex::encode(tx_id));

        // Route around suspected censors
        if self.suspected_censors.contains(&target_node) {
            let route_around = CountermeasureStrategy::RouteAround { 
                avoid_nodes: self.suspected_censors.clone() 
            };
            self.countermeasures.push(route_around);
        }

        // Apply transaction disguise
        let disguise_level = match self.resistance_level {
            CensorshipResistance::Active => 2,
            CensorshipResistance::QuantumEnhanced => 5,
            _ => 1,
        };
        
        let disguise = CountermeasureStrategy::TransactionDisguise { 
            obfuscation_level: disguise_level 
        };
        self.countermeasures.push(disguise);

        // Quantum evasion for highest resistance level
        if matches!(self.resistance_level, CensorshipResistance::QuantumEnhanced) {
            let quantum_evasion = CountermeasureStrategy::QuantumEvasion { 
                randomization_strength: 0.8 
            };
            self.countermeasures.push(quantum_evasion);
        }

        // Apply reputation penalty
        let penalty = CountermeasureStrategy::ReputationPenalty { 
            penalty_factor: 0.1 
        };
        self.countermeasures.push(penalty);

        self.detection_metrics.countermeasures_applied += 1;
        
        debug!("Applied {} countermeasures", self.countermeasures.len());
        Ok(())
    }

    /// Analyze patterns to detect censorship
    async fn analyze_censorship_patterns(&mut self, tx_id: &TransactionId, from_node: NodeId) -> Result<bool> {
        // Track rejection patterns
        let pattern = self.rejection_patterns.entry(from_node).or_insert_with(|| RejectionPattern {
            total_rejections: 0,
            recent_rejections: Vec::new(),
            rejection_rate: 0.0,
            suspicious_patterns: Vec::new(),
        });

        // Simple heuristic: check for high rejection rate
        let now = Instant::now();
        pattern.recent_rejections.retain(|&time| now.duration_since(time) < Duration::from_secs(300)); // 5 minute window
        pattern.recent_rejections.push(now);
        pattern.total_rejections += 1;

        // Calculate rejection rate
        pattern.rejection_rate = pattern.recent_rejections.len() as f64 / 300.0; // per second

        // Detect suspicious patterns
        if pattern.rejection_rate > 0.1 { // More than 0.1 rejections per second
            let suspicious = SuspiciousPattern::HighRejectionRate {
                rate: pattern.rejection_rate,
                window: Duration::from_secs(300),
            };
            
            if !pattern.suspicious_patterns.iter().any(|p| matches!(p, SuspiciousPattern::HighRejectionRate { .. })) {
                pattern.suspicious_patterns.push(suspicious);
            }
            
            return Ok(true);
        }

        // Check for selective targeting (simplified)
        if pattern.total_rejections > 50 && pattern.rejection_rate > 0.05 {
            let selective = SuspiciousPattern::SelectiveTargeting {
                target_nodes: HashSet::from([from_node]),
                bias: pattern.rejection_rate * 10.0,
            };
            pattern.suspicious_patterns.push(selective);
            return Ok(true);
        }

        Ok(false)
    }

    /// Check if evidence is strong enough to mark as censor
    async fn is_strong_censorship_evidence(&self, node: NodeId) -> Result<bool> {
        if let Some(pattern) = self.rejection_patterns.get(&node) {
            // Strong evidence criteria:
            // 1. High total rejections
            // 2. Sustained high rejection rate
            // 3. Multiple suspicious patterns
            
            let high_total = pattern.total_rejections > 100;
            let high_rate = pattern.rejection_rate > 0.2;
            let multiple_patterns = pattern.suspicious_patterns.len() > 1;
            
            Ok(high_total && high_rate && multiple_patterns)
        } else {
            Ok(false)
        }
    }

    /// Get list of suspected censoring nodes
    pub fn get_suspected_censors(&self) -> &HashSet<NodeId> {
        &self.suspected_censors
    }

    /// Get censorship detection metrics
    pub fn get_metrics(&self) -> &CensorshipMetrics {
        &self.detection_metrics
    }

    /// Get active countermeasures
    pub fn get_active_countermeasures(&self) -> &Vec<CountermeasureStrategy> {
        &self.countermeasures
    }

    /// Clear old countermeasures
    pub fn clear_old_countermeasures(&mut self) {
        // Keep only recent countermeasures
        if self.countermeasures.len() > 100 {
            self.countermeasures.drain(0..50);
        }
    }

    /// Update resistance level
    pub fn set_resistance_level(&mut self, level: CensorshipResistance) {
        self.resistance_level = level;
        info!("Anti-censorship resistance level updated to {:?}", level);
    }

    /// Generate anti-censorship report
    pub async fn generate_report(&self) -> Result<CensorshipReport> {
        let report = CensorshipReport {
            resistance_level: self.resistance_level,
            suspected_censors_count: self.suspected_censors.len(),
            total_censorship_events: self.detection_metrics.total_censorship_events,
            countermeasures_applied: self.detection_metrics.countermeasures_applied,
            success_rate: self.detection_metrics.success_rate,
            most_suspicious_nodes: self.get_most_suspicious_nodes().await?,
        };
        
        Ok(report)
    }

    /// Get most suspicious nodes for reporting
    async fn get_most_suspicious_nodes(&self) -> Result<Vec<(NodeId, f64)>> {
        let mut suspicious: Vec<(NodeId, f64)> = self.rejection_patterns
            .iter()
            .map(|(&node_id, pattern)| {
                let suspicion_score = pattern.rejection_rate * 
                    (pattern.suspicious_patterns.len() as f64) * 
                    (pattern.total_rejections as f64).log2();
                (node_id, suspicion_score)
            })
            .collect();
        
        suspicious.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        suspicious.truncate(10); // Top 10 most suspicious
        
        Ok(suspicious)
    }
}

/// Censorship resistance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CensorshipReport {
    pub resistance_level: CensorshipResistance,
    pub suspected_censors_count: usize,
    pub total_censorship_events: u64,
    pub countermeasures_applied: u64,
    pub success_rate: f64,
    pub most_suspicious_nodes: Vec<(NodeId, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_anti_censorship_manager_creation() {
        let manager = AntiCensorshipManager::new(true).unwrap();
        
        assert!(matches!(manager.resistance_level, CensorshipResistance::Active));
        assert!(manager.suspected_censors.is_empty());
    }

    #[tokio::test]
    async fn test_censorship_detection() {
        let mut manager = AntiCensorshipManager::new(true).unwrap();
        let tx_id = Uuid::new_v4().into_bytes();
        let node_id = [1u8; 32];

        // First few attempts should not trigger
        for _ in 0..5 {
            let is_censorship = manager.is_censorship_attempt(&tx_id, node_id).await.unwrap();
            assert!(!is_censorship);
        }

        // Many rapid rejections should trigger detection
        for _ in 0..60 {
            let _is_censorship = manager.is_censorship_attempt(&tx_id, node_id).await.unwrap();
        }

        // Should now detect censorship
        let is_censorship = manager.is_censorship_attempt(&tx_id, node_id).await.unwrap();
        assert!(is_censorship);
    }

    #[tokio::test]
    async fn test_countermeasures() {
        let mut manager = AntiCensorshipManager::new(true).unwrap();
        let tx_id = Uuid::new_v4().into_bytes();
        let node_id = [1u8; 32];

        manager.apply_countermeasures(&tx_id, node_id).await.unwrap();
        
        assert!(!manager.countermeasures.is_empty());
        assert!(manager.detection_metrics.countermeasures_applied > 0);
    }
}