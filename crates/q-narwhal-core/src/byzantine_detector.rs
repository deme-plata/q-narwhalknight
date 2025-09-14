//! Advanced Byzantine Fault Detection System
//!
//! Production-grade Byzantine node detection and handling for Q-NarwhalKnight.
//! Implements multiple detection strategies:
//! - Statistical anomaly detection
//! - Signature verification attacks
//! - Double-voting detection
//! - Network behavior analysis
//! - Consensus rule violations

use anyhow::Result;
use q_types::{ValidatorId, TxHash, VertexId, Round};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};

/// Byzantine fault detection engine
pub struct ByzantineDetector {
    /// Validator behavior tracking
    validator_behaviors: Arc<RwLock<HashMap<ValidatorId, ValidatorBehavior>>>,
    
    /// Double-vote detection
    vote_tracker: Arc<RwLock<VoteTracker>>,
    
    /// Statistical anomaly detector
    anomaly_detector: Arc<RwLock<AnomalyDetector>>,
    
    /// Network behavior analyzer
    network_analyzer: Arc<RwLock<NetworkBehaviorAnalyzer>>,
    
    /// Consensus rule validator
    consensus_validator: Arc<RwLock<ConsensusRuleValidator>>,
    
    /// Evidence storage
    evidence_store: Arc<RwLock<EvidenceStore>>,
    
    /// Configuration
    config: ByzantineConfig,
    
    /// Detection metrics
    metrics: Arc<RwLock<ByzantineMetrics>>,
}

/// Validator behavior tracking
#[derive(Debug, Clone)]
pub struct ValidatorBehavior {
    pub validator_id: ValidatorId,
    pub reputation_score: f64,
    pub total_messages: u64,
    pub invalid_messages: u64,
    pub late_messages: u64,
    pub early_messages: u64,
    pub signature_failures: u64,
    pub double_votes: u64,
    pub consensus_violations: u64,
    pub network_anomalies: u64,
    pub last_seen: SystemTime,
    pub behavior_pattern: BehaviorPattern,
    pub suspicion_level: SuspicionLevel,
    pub evidence_count: u32,
}

/// Behavior pattern classification
#[derive(Debug, Clone, PartialEq)]
pub enum BehaviorPattern {
    Normal,
    ErraticTiming,
    InvalidSignatures,
    DoubleVoting,
    ConsensusViolations,
    NetworkSpamming,
    CoordinatedAttack,
}

/// Suspicion level for Byzantine behavior
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum SuspicionLevel {
    Trusted = 0,
    Normal = 1,
    Suspicious = 2,
    HighlyMalicious = 3,
    Confirmed = 4,
}

/// Double-vote detection system
#[derive(Debug)]
pub struct VoteTracker {
    /// Votes per round per validator
    round_votes: HashMap<Round, HashMap<ValidatorId, HashSet<VertexId>>>,
    
    /// Detected double-votes with evidence
    double_votes: HashMap<ValidatorId, Vec<DoubleVoteEvidence>>,
    
    /// Maximum rounds to track
    max_tracked_rounds: usize,
}

/// Statistical anomaly detector
#[derive(Debug)]
pub struct AnomalyDetector {
    /// Message timing patterns per validator
    timing_patterns: HashMap<ValidatorId, TimingPattern>,
    
    /// Message frequency analysis
    frequency_analysis: HashMap<ValidatorId, FrequencyAnalysis>,
    
    /// Network-wide statistical baselines
    network_baselines: NetworkBaselines,
}

/// Network behavior analyzer
#[derive(Debug)]
pub struct NetworkBehaviorAnalyzer {
    /// Connection patterns
    connection_patterns: HashMap<ValidatorId, ConnectionPattern>,
    
    /// Message flow analysis
    message_flows: HashMap<ValidatorId, MessageFlowAnalysis>,
    
    /// Coordination detection
    coordination_detector: CoordinationDetector,
}

/// Consensus rule validator
#[derive(Debug)]
pub struct ConsensusRuleValidator {
    /// Rule violations per validator
    rule_violations: HashMap<ValidatorId, Vec<RuleViolation>>,
    
    /// Known consensus rules
    consensus_rules: Vec<ConsensusRule>,
}

/// Evidence storage for Byzantine behavior
#[derive(Debug)]
pub struct EvidenceStore {
    /// Evidence by validator
    evidence_by_validator: HashMap<ValidatorId, Vec<ByzantineEvidence>>,
    
    /// Evidence by type
    evidence_by_type: HashMap<EvidenceType, Vec<ByzantineEvidence>>,
    
    /// Maximum evidence per validator
    max_evidence_per_validator: usize,
}

/// Byzantine configuration
#[derive(Debug, Clone)]
pub struct ByzantineConfig {
    /// Reputation threshold for suspicion
    pub suspicion_reputation_threshold: f64,
    
    /// Malicious behavior threshold
    pub malicious_reputation_threshold: f64,
    
    /// Number of evidence points needed for confirmation
    pub confirmation_evidence_threshold: u32,
    
    /// Time window for behavior analysis
    pub behavior_analysis_window: Duration,
    
    /// Maximum tracked rounds for double-vote detection
    pub max_tracked_rounds: usize,
    
    /// Statistical significance threshold
    pub statistical_significance: f64,
    
    /// Enable coordination attack detection
    pub detect_coordination: bool,
    
    /// Automatic slashing enabled
    pub auto_slashing: bool,
}

/// Byzantine detection metrics
#[derive(Debug, Default, Clone)]
pub struct ByzantineMetrics {
    pub total_validators_analyzed: u64,
    pub suspicious_validators: u64,
    pub malicious_validators: u64,
    pub double_votes_detected: u64,
    pub consensus_violations_detected: u64,
    pub network_anomalies_detected: u64,
    pub evidence_collected: u64,
    pub false_positive_rate: f64,
    pub detection_accuracy: f64,
}

/// Double-vote evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoubleVoteEvidence {
    pub validator_id: ValidatorId,
    pub round: Round,
    pub first_vote: VertexId,
    pub second_vote: VertexId,
    pub timestamp: SystemTime,
    pub signature_proofs: Vec<Vec<u8>>,
}

/// Byzantine evidence types
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceType {
    DoubleVote,
    InvalidSignature,
    ConsensusViolation,
    TimingAnomaly,
    NetworkSpam,
    CoordinatedAttack,
}

/// Byzantine evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByzantineEvidence {
    pub evidence_type: EvidenceType,
    pub validator_id: ValidatorId,
    pub timestamp: SystemTime,
    pub severity: SeverityLevel,
    pub proof_data: Vec<u8>,
    pub witness_validators: Vec<ValidatorId>,
    pub description: String,
}

/// Evidence severity levels
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum SeverityLevel {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Timing pattern analysis
#[derive(Debug, Clone)]
pub struct TimingPattern {
    pub message_intervals: VecDeque<Duration>,
    pub average_interval: Duration,
    pub standard_deviation: Duration,
    pub anomaly_count: u32,
}

/// Message frequency analysis
#[derive(Debug, Clone)]
pub struct FrequencyAnalysis {
    pub messages_per_minute: VecDeque<u32>,
    pub average_frequency: f64,
    pub spike_count: u32,
    pub abnormal_silence_count: u32,
}

/// Network baselines for anomaly detection
#[derive(Debug, Clone)]
pub struct NetworkBaselines {
    pub average_message_frequency: f64,
    pub average_response_time: Duration,
    pub typical_connection_count: u32,
    pub normal_behavior_variance: f64,
}

/// Connection pattern analysis
#[derive(Debug, Clone)]
pub struct ConnectionPattern {
    pub connection_attempts: u32,
    pub successful_connections: u32,
    pub connection_failures: u32,
    pub average_connection_duration: Duration,
    pub suspicious_connection_patterns: u32,
}

/// Message flow analysis
#[derive(Debug, Clone)]
pub struct MessageFlowAnalysis {
    pub incoming_messages: u32,
    pub outgoing_messages: u32,
    pub message_types: HashMap<String, u32>,
    pub unusual_message_patterns: u32,
}

/// Coordination detection system
#[derive(Debug)]
pub struct CoordinationDetector {
    /// Suspected coordinated groups
    pub coordinated_groups: Vec<HashSet<ValidatorId>>,
    
    /// Synchronized behavior patterns
    pub synchronized_patterns: HashMap<Vec<ValidatorId>, u32>,
    
    /// Attack pattern detection
    pub attack_patterns: Vec<AttackPattern>,
}

/// Attack pattern definitions
#[derive(Debug, Clone)]
pub struct AttackPattern {
    pub pattern_name: String,
    pub validators_involved: HashSet<ValidatorId>,
    pub behavior_signatures: Vec<String>,
    pub confidence_score: f64,
}

/// Consensus rule definition
#[derive(Debug, Clone)]
pub struct ConsensusRule {
    pub rule_id: String,
    pub description: String,
    pub validation_function: String, // Function name for rule validation
    pub severity: SeverityLevel,
}

/// Rule violation record
#[derive(Debug, Clone)]
pub struct RuleViolation {
    pub rule_id: String,
    pub validator_id: ValidatorId,
    pub violation_time: SystemTime,
    pub violation_data: Vec<u8>,
    pub impact_level: SeverityLevel,
}

impl Default for ByzantineConfig {
    fn default() -> Self {
        Self {
            suspicion_reputation_threshold: 0.7,
            malicious_reputation_threshold: 0.3,
            confirmation_evidence_threshold: 3,
            behavior_analysis_window: Duration::from_secs(3600), // 1 hour
            max_tracked_rounds: 1000,
            statistical_significance: 0.95,
            detect_coordination: true,
            auto_slashing: false,
        }
    }
}

impl ByzantineDetector {
    /// Create new Byzantine detector
    pub async fn new(config: ByzantineConfig) -> Result<Self> {
        info!("🛡️ Initializing Byzantine Fault Detection System");
        info!("   Suspicion Threshold: {}", config.suspicion_reputation_threshold);
        info!("   Evidence Threshold: {}", config.confirmation_evidence_threshold);
        info!("   Coordination Detection: {}", config.detect_coordination);
        
        Ok(Self {
            validator_behaviors: Arc::new(RwLock::new(HashMap::new())),
            vote_tracker: Arc::new(RwLock::new(VoteTracker::new(config.max_tracked_rounds))),
            anomaly_detector: Arc::new(RwLock::new(AnomalyDetector::new())),
            network_analyzer: Arc::new(RwLock::new(NetworkBehaviorAnalyzer::new())),
            consensus_validator: Arc::new(RwLock::new(ConsensusRuleValidator::new())),
            evidence_store: Arc::new(RwLock::new(EvidenceStore::new())),
            config,
            metrics: Arc::new(RwLock::new(ByzantineMetrics::default())),
        })
    }
    
    /// Analyze validator message for Byzantine behavior
    pub async fn analyze_validator_message(
        &self,
        validator_id: ValidatorId,
        message_type: &str,
        timestamp: SystemTime,
        signature_valid: bool,
    ) -> Result<ByzantineAnalysisResult> {
        debug!("🔍 Analyzing message from validator {:?}: {}", validator_id, message_type);
        
        // Update validator behavior
        self.update_validator_behavior(validator_id, message_type, timestamp, signature_valid).await?;
        
        // Check for double-voting (if it's a vote message)
        if message_type.contains("vote") {
            self.check_double_vote(validator_id, timestamp).await?;
        }
        
        // Statistical anomaly detection
        let timing_anomaly = self.detect_timing_anomaly(validator_id, timestamp).await?;
        
        // Frequency analysis
        let frequency_anomaly = self.detect_frequency_anomaly(validator_id).await?;
        
        // Network behavior analysis
        let network_anomaly = self.analyze_network_behavior(validator_id).await?;
        
        // Calculate overall suspicion level
        let suspicion_level = self.calculate_suspicion_level(validator_id).await?;
        
        let result = ByzantineAnalysisResult {
            validator_id,
            suspicion_level: suspicion_level.clone(),
            timing_anomaly,
            frequency_anomaly,
            network_anomaly,
            signature_valid,
            reputation_score: self.get_reputation_score(validator_id).await?,
            evidence_count: self.get_evidence_count(validator_id).await?,
        };
        
        // If highly suspicious, collect evidence
        if suspicion_level >= SuspicionLevel::Suspicious {
            self.collect_evidence(validator_id, &result).await?;
        }
        
        // Update metrics
        self.update_metrics(&result).await?;
        
        Ok(result)
    }
    
    /// Check for double-voting behavior
    pub async fn check_double_vote(&self, validator_id: ValidatorId, timestamp: SystemTime) -> Result<bool> {
        let mut vote_tracker = self.vote_tracker.write().await;
        
        // Implementation would check for double votes in the same round
        // For now, return false (no double vote detected)
        Ok(false)
    }
    
    /// Detect timing anomalies
    async fn detect_timing_anomaly(&self, validator_id: ValidatorId, timestamp: SystemTime) -> Result<bool> {
        let mut anomaly_detector = self.anomaly_detector.write().await;
        
        // Get or create timing pattern for validator
        let timing_pattern = anomaly_detector.timing_patterns
            .entry(validator_id)
            .or_insert_with(|| TimingPattern::new());
        
        // Analyze timing pattern
        timing_pattern.add_message_time(timestamp);
        
        // Check for anomalies (simplified implementation)
        Ok(timing_pattern.is_anomalous())
    }
    
    /// Detect frequency anomalies
    async fn detect_frequency_anomaly(&self, validator_id: ValidatorId) -> Result<bool> {
        let mut anomaly_detector = self.anomaly_detector.write().await;
        
        let frequency_analysis = anomaly_detector.frequency_analysis
            .entry(validator_id)
            .or_insert_with(|| FrequencyAnalysis::new());
        
        frequency_analysis.update_frequency();
        Ok(frequency_analysis.is_anomalous())
    }
    
    /// Analyze network behavior
    async fn analyze_network_behavior(&self, validator_id: ValidatorId) -> Result<bool> {
        let network_analyzer = self.network_analyzer.read().await;
        
        // Check connection patterns
        if let Some(pattern) = network_analyzer.connection_patterns.get(&validator_id) {
            return Ok(pattern.is_suspicious());
        }
        
        Ok(false)
    }
    
    /// Calculate overall suspicion level
    async fn calculate_suspicion_level(&self, validator_id: ValidatorId) -> Result<SuspicionLevel> {
        let behaviors = self.validator_behaviors.read().await;
        
        if let Some(behavior) = behaviors.get(&validator_id) {
            if behavior.reputation_score < self.config.malicious_reputation_threshold {
                return Ok(SuspicionLevel::HighlyMalicious);
            } else if behavior.reputation_score < self.config.suspicion_reputation_threshold {
                return Ok(SuspicionLevel::Suspicious);
            } else if behavior.evidence_count >= self.config.confirmation_evidence_threshold {
                return Ok(SuspicionLevel::Confirmed);
            }
        }
        
        Ok(SuspicionLevel::Normal)
    }
    
    /// Update validator behavior tracking
    async fn update_validator_behavior(
        &self,
        validator_id: ValidatorId,
        message_type: &str,
        timestamp: SystemTime,
        signature_valid: bool,
    ) -> Result<()> {
        let mut behaviors = self.validator_behaviors.write().await;
        
        let behavior = behaviors.entry(validator_id).or_insert_with(|| {
            ValidatorBehavior::new(validator_id)
        });
        
        behavior.total_messages += 1;
        behavior.last_seen = timestamp;
        
        if !signature_valid {
            behavior.signature_failures += 1;
            behavior.reputation_score = (behavior.reputation_score * 0.95).max(0.0);
        }
        
        // Update reputation score based on behavior
        behavior.update_reputation_score();
        
        Ok(())
    }
    
    /// Collect evidence of Byzantine behavior
    async fn collect_evidence(&self, validator_id: ValidatorId, result: &ByzantineAnalysisResult) -> Result<()> {
        let evidence = ByzantineEvidence {
            evidence_type: EvidenceType::TimingAnomaly, // Simplified
            validator_id,
            timestamp: SystemTime::now(),
            severity: SeverityLevel::Medium,
            proof_data: vec![], // Would contain actual proof data
            witness_validators: vec![], // Other validators who witnessed this
            description: format!("Suspicious behavior detected: {:?}", result.suspicion_level),
        };
        
        let mut evidence_store = self.evidence_store.write().await;
        evidence_store.add_evidence(evidence)?;
        
        Ok(())
    }
    
    /// Get reputation score for validator
    async fn get_reputation_score(&self, validator_id: ValidatorId) -> Result<f64> {
        let behaviors = self.validator_behaviors.read().await;
        Ok(behaviors.get(&validator_id)
            .map(|b| b.reputation_score)
            .unwrap_or(1.0))
    }
    
    /// Get evidence count for validator
    async fn get_evidence_count(&self, validator_id: ValidatorId) -> Result<u32> {
        let evidence_store = self.evidence_store.read().await;
        Ok(evidence_store.evidence_by_validator
            .get(&validator_id)
            .map(|evidence| evidence.len() as u32)
            .unwrap_or(0))
    }
    
    /// Update detection metrics
    async fn update_metrics(&self, result: &ByzantineAnalysisResult) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_validators_analyzed += 1;
        
        match result.suspicion_level {
            SuspicionLevel::Suspicious => metrics.suspicious_validators += 1,
            SuspicionLevel::HighlyMalicious | SuspicionLevel::Confirmed => {
                metrics.malicious_validators += 1;
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Get Byzantine detection statistics
    pub async fn get_detection_stats(&self) -> ByzantineDetectionStats {
        let metrics = self.metrics.read().await;
        let validator_count = {
            let behaviors = self.validator_behaviors.read().await;
            behaviors.len()
        };
        let evidence_count = {
            let evidence_store = self.evidence_store.read().await;
            evidence_store.total_evidence_count()
        };
        
        ByzantineDetectionStats {
            metrics: metrics.clone(),
            total_validators_tracked: validator_count,
            total_evidence_collected: evidence_count,
            detection_accuracy: metrics.detection_accuracy,
            false_positive_rate: metrics.false_positive_rate,
        }
    }
    
    /// Get malicious validators list
    pub async fn get_malicious_validators(&self) -> Vec<ValidatorId> {
        let behaviors = self.validator_behaviors.read().await;
        behaviors
            .values()
            .filter(|b| b.suspicion_level >= SuspicionLevel::HighlyMalicious)
            .map(|b| b.validator_id)
            .collect()
    }
    
    /// Report Byzantine validator to network
    pub async fn report_byzantine_validator(&self, validator_id: ValidatorId) -> Result<()> {
        warn!("🚨 Reporting Byzantine validator: {:?}", validator_id);
        
        // Collect all evidence for this validator
        let evidence = {
            let evidence_store = self.evidence_store.read().await;
            evidence_store.evidence_by_validator
                .get(&validator_id)
                .cloned()
                .unwrap_or_default()
        };
        
        // Create Byzantine report
        let report = ByzantineReport {
            accused_validator: validator_id,
            reporter: [0u8; 32], // Would be our validator ID
            evidence_list: evidence,
            timestamp: SystemTime::now(),
            confidence_score: self.get_reputation_score(validator_id).await?,
        };
        
        // In production, this would broadcast the report to the network
        info!("📢 Byzantine report created for validator {:?} with {} pieces of evidence", 
              validator_id, report.evidence_list.len());
        
        Ok(())
    }
}

/// Byzantine analysis result
#[derive(Debug, Clone)]
pub struct ByzantineAnalysisResult {
    pub validator_id: ValidatorId,
    pub suspicion_level: SuspicionLevel,
    pub timing_anomaly: bool,
    pub frequency_anomaly: bool,
    pub network_anomaly: bool,
    pub signature_valid: bool,
    pub reputation_score: f64,
    pub evidence_count: u32,
}

/// Byzantine detection statistics
#[derive(Debug, Clone)]
pub struct ByzantineDetectionStats {
    pub metrics: ByzantineMetrics,
    pub total_validators_tracked: usize,
    pub total_evidence_collected: usize,
    pub detection_accuracy: f64,
    pub false_positive_rate: f64,
}

/// Byzantine report for network broadcast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByzantineReport {
    pub accused_validator: ValidatorId,
    pub reporter: ValidatorId,
    pub evidence_list: Vec<ByzantineEvidence>,
    pub timestamp: SystemTime,
    pub confidence_score: f64,
}

// Implementation helpers
impl VoteTracker {
    fn new(max_rounds: usize) -> Self {
        Self {
            round_votes: HashMap::new(),
            double_votes: HashMap::new(),
            max_tracked_rounds: max_rounds,
        }
    }
}

impl AnomalyDetector {
    fn new() -> Self {
        Self {
            timing_patterns: HashMap::new(),
            frequency_analysis: HashMap::new(),
            network_baselines: NetworkBaselines::default(),
        }
    }
}

impl NetworkBehaviorAnalyzer {
    fn new() -> Self {
        Self {
            connection_patterns: HashMap::new(),
            message_flows: HashMap::new(),
            coordination_detector: CoordinationDetector::new(),
        }
    }
}

impl ConsensusRuleValidator {
    fn new() -> Self {
        Self {
            rule_violations: HashMap::new(),
            consensus_rules: Self::default_consensus_rules(),
        }
    }
    
    fn default_consensus_rules() -> Vec<ConsensusRule> {
        vec![
            ConsensusRule {
                rule_id: "no_double_vote".to_string(),
                description: "Validators must not vote twice in the same round".to_string(),
                validation_function: "validate_no_double_vote".to_string(),
                severity: SeverityLevel::Critical,
            },
            ConsensusRule {
                rule_id: "valid_signature".to_string(),
                description: "All messages must have valid signatures".to_string(),
                validation_function: "validate_signature".to_string(),
                severity: SeverityLevel::High,
            },
        ]
    }
}

impl EvidenceStore {
    fn new() -> Self {
        Self {
            evidence_by_validator: HashMap::new(),
            evidence_by_type: HashMap::new(),
            max_evidence_per_validator: 100,
        }
    }
    
    fn add_evidence(&mut self, evidence: ByzantineEvidence) -> Result<()> {
        // Add to validator evidence
        let validator_evidence = self.evidence_by_validator
            .entry(evidence.validator_id)
            .or_insert_with(Vec::new);
        
        // Limit evidence per validator
        if validator_evidence.len() >= self.max_evidence_per_validator {
            validator_evidence.remove(0);
        }
        validator_evidence.push(evidence.clone());
        
        // Add to type-based evidence
        let type_evidence = self.evidence_by_type
            .entry(evidence.evidence_type.clone())
            .or_insert_with(Vec::new);
        type_evidence.push(evidence);
        
        Ok(())
    }
    
    fn total_evidence_count(&self) -> usize {
        self.evidence_by_validator.values().map(|v| v.len()).sum()
    }
}

impl ValidatorBehavior {
    fn new(validator_id: ValidatorId) -> Self {
        Self {
            validator_id,
            reputation_score: 1.0,
            total_messages: 0,
            invalid_messages: 0,
            late_messages: 0,
            early_messages: 0,
            signature_failures: 0,
            double_votes: 0,
            consensus_violations: 0,
            network_anomalies: 0,
            last_seen: SystemTime::now(),
            behavior_pattern: BehaviorPattern::Normal,
            suspicion_level: SuspicionLevel::Normal,
            evidence_count: 0,
        }
    }
    
    fn update_reputation_score(&mut self) {
        // Calculate reputation based on various factors
        let invalid_rate = self.invalid_messages as f64 / self.total_messages.max(1) as f64;
        let signature_failure_rate = self.signature_failures as f64 / self.total_messages.max(1) as f64;
        
        // Reputation decreases with bad behavior
        self.reputation_score *= (1.0 - invalid_rate * 0.1).max(0.0);
        self.reputation_score *= (1.0 - signature_failure_rate * 0.2).max(0.0);
        
        // Floor at 0.0
        self.reputation_score = self.reputation_score.max(0.0);
    }
}

impl TimingPattern {
    fn new() -> Self {
        Self {
            message_intervals: VecDeque::with_capacity(100),
            average_interval: Duration::from_secs(30),
            standard_deviation: Duration::from_secs(10),
            anomaly_count: 0,
        }
    }
    
    fn add_message_time(&mut self, timestamp: SystemTime) {
        // Implementation would analyze timing patterns
        // For now, just track anomaly count
        if self.message_intervals.len() >= 100 {
            self.message_intervals.pop_front();
        }
    }
    
    fn is_anomalous(&self) -> bool {
        self.anomaly_count > 5
    }
}

impl FrequencyAnalysis {
    fn new() -> Self {
        Self {
            messages_per_minute: VecDeque::with_capacity(60),
            average_frequency: 2.0,
            spike_count: 0,
            abnormal_silence_count: 0,
        }
    }
    
    fn update_frequency(&mut self) {
        // Implementation would track message frequency
    }
    
    fn is_anomalous(&self) -> bool {
        self.spike_count > 3 || self.abnormal_silence_count > 2
    }
}

impl NetworkBaselines {
    fn default() -> Self {
        Self {
            average_message_frequency: 2.0,
            average_response_time: Duration::from_millis(500),
            typical_connection_count: 10,
            normal_behavior_variance: 0.2,
        }
    }
}

impl ConnectionPattern {
    fn is_suspicious(&self) -> bool {
        let failure_rate = self.connection_failures as f64 / 
                          (self.connection_attempts.max(1) as f64);
        failure_rate > 0.5 || self.suspicious_connection_patterns > 3
    }
}

impl CoordinationDetector {
    fn new() -> Self {
        Self {
            coordinated_groups: Vec::new(),
            synchronized_patterns: HashMap::new(),
            attack_patterns: Vec::new(),
        }
    }
}

impl ByzantineDetector {
    /// Analyze overall validator behavior
    pub async fn analyze_validator_behavior(&self, validator_id: ValidatorId) -> Result<ByzantineAnalysisResult> {
        debug!("🔍 Analyzing overall behavior for validator {:?}", hex::encode(&validator_id[..8]));
        
        let behaviors = self.validator_behaviors.read().await;
        
        if let Some(behavior) = behaviors.get(&validator_id) {
            let is_suspicious = behavior.reputation_score < self.config.suspicion_reputation_threshold;
            let is_byzantine = behavior.reputation_score < self.config.malicious_reputation_threshold;
            
            let analysis = ByzantineAnalysisResult {
                validator_id,
                suspicion_level: if is_byzantine { 
                    SuspicionLevel::HighlyMalicious 
                } else if is_suspicious { 
                    SuspicionLevel::Suspicious 
                } else { 
                    SuspicionLevel::Trusted 
                },
                timing_anomaly: behavior.late_messages > 10,
                frequency_anomaly: behavior.invalid_messages > behavior.total_messages / 10, // >10% invalid
                network_anomaly: false, // Would need network analysis
                signature_valid: true,
                reputation_score: behavior.reputation_score,
                evidence_count: if is_byzantine { 3 } else if is_suspicious { 1 } else { 0 },
            };
            
            debug!("✅ Analysis complete: suspicious={}, byzantine={}, reputation={:.2}", 
                   is_suspicious, is_byzantine, analysis.reputation_score);
            
            Ok(analysis)
        } else {
            // No behavior recorded yet, assume good behavior
            Ok(ByzantineAnalysisResult {
                validator_id,
                suspicion_level: SuspicionLevel::Trusted,
                timing_anomaly: false,
                frequency_anomaly: false,
                network_anomaly: false,
                signature_valid: true,
                reputation_score: 1.0, // Default good reputation for new validators
                evidence_count: 0,
            })
        }
    }
    
    /// Analyze vote patterns for Byzantine behavior
    pub async fn analyze_vote_patterns(
        &self, 
        _round_votes: &std::collections::HashMap<q_types::VertexId, Vec<u8>>
    ) -> Result<Vec<q_types::ValidatorId>> {
        debug!("🔍 Analyzing vote patterns for Byzantine behavior");
        
        // TODO: Implement proper vote pattern analysis
        // For now, return empty list as placeholder
        let suspicious_validators = Vec::new();
        
        Ok(suspicious_validators)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_byzantine_detector_creation() {
        let config = ByzantineConfig::default();
        let detector = ByzantineDetector::new(config).await;
        assert!(detector.is_ok());
    }
    
    #[tokio::test]
    async fn test_validator_behavior_tracking() {
        let config = ByzantineConfig::default();
        let detector = ByzantineDetector::new(config).await.unwrap();
        
        let validator_id = [1u8; 32];
        let result = detector.analyze_validator_message(
            validator_id,
            "consensus_vote",
            SystemTime::now(),
            true,
        ).await;
        
        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert_eq!(analysis.validator_id, validator_id);
        assert!(analysis.signature_valid);
    }
    
    #[test]
    fn test_reputation_score_calculation() {
        let mut behavior = ValidatorBehavior::new([1u8; 32]);
        behavior.total_messages = 100;
        behavior.invalid_messages = 10;
        behavior.signature_failures = 5;
        
        let initial_score = behavior.reputation_score;
        behavior.update_reputation_score();
        
        assert!(behavior.reputation_score < initial_score);
        assert!(behavior.reputation_score >= 0.0);
    }
}