/// Anonymity Set Monitoring for Q-NarwhalKnight
///
/// This module provides real-time anonymity risk assessment by monitoring
/// various factors that could potentially deanonymize users:
///
/// - Guard node usage patterns
/// - Circuit reuse frequency
/// - Traffic volume correlation
/// - Timing patterns
/// - Peer connection diversity
/// - Geographic distribution of relays
///
/// The goal is to provide actionable intelligence to the privacy layer
/// about when anonymity might be compromised and recommendations for
/// protective actions.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    net::IpAddr,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Anonymity risk level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RiskLevel {
    /// No known anonymity concerns
    Minimal,
    /// Minor concerns, continue with caution
    Low,
    /// Moderate risk, consider protective measures
    Moderate,
    /// High risk, take immediate protective action
    High,
    /// Critical risk, stop operations immediately
    Critical,
}

impl RiskLevel {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            RiskLevel::Minimal => "Minimal",
            RiskLevel::Low => "Low",
            RiskLevel::Moderate => "Moderate",
            RiskLevel::High => "High",
            RiskLevel::Critical => "Critical",
        }
    }

    /// Get color for display
    pub fn color(&self) -> &'static str {
        match self {
            RiskLevel::Minimal => "green",
            RiskLevel::Low => "yellow",
            RiskLevel::Moderate => "orange",
            RiskLevel::High => "red",
            RiskLevel::Critical => "magenta",
        }
    }

    /// Get numeric score (0-100)
    pub fn score(&self) -> u8 {
        match self {
            RiskLevel::Minimal => 10,
            RiskLevel::Low => 30,
            RiskLevel::Moderate => 50,
            RiskLevel::High => 75,
            RiskLevel::Critical => 95,
        }
    }
}

/// Type of anonymity threat
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThreatType {
    /// Guard discovery attack
    GuardDiscovery,
    /// Traffic correlation attack
    TrafficCorrelation,
    /// Timing analysis attack
    TimingAnalysis,
    /// Sybil attack (malicious nodes)
    SybilAttack,
    /// Geographic deanonymization
    GeographicCorrelation,
    /// Circuit linking attack
    CircuitLinking,
    /// Bandwidth fingerprinting
    BandwidthFingerprinting,
    /// Long-term intersection attack
    IntersectionAttack,
    /// Website fingerprinting
    WebsiteFingerprinting,
    /// Relay enumeration
    RelayEnumeration,
}

impl ThreatType {
    pub fn name(&self) -> &'static str {
        match self {
            ThreatType::GuardDiscovery => "Guard Discovery",
            ThreatType::TrafficCorrelation => "Traffic Correlation",
            ThreatType::TimingAnalysis => "Timing Analysis",
            ThreatType::SybilAttack => "Sybil Attack",
            ThreatType::GeographicCorrelation => "Geographic Correlation",
            ThreatType::CircuitLinking => "Circuit Linking",
            ThreatType::BandwidthFingerprinting => "Bandwidth Fingerprinting",
            ThreatType::IntersectionAttack => "Intersection Attack",
            ThreatType::WebsiteFingerprinting => "Website Fingerprinting",
            ThreatType::RelayEnumeration => "Relay Enumeration",
        }
    }

    pub fn mitigation(&self) -> &'static str {
        match self {
            ThreatType::GuardDiscovery => "Enable vanguards, rotate guards less frequently",
            ThreatType::TrafficCorrelation => "Enable traffic shaping, add padding",
            ThreatType::TimingAnalysis => "Enable timing obfuscation, add jitter",
            ThreatType::SybilAttack => "Use trusted guards, verify relay diversity",
            ThreatType::GeographicCorrelation => "Distribute circuits across regions",
            ThreatType::CircuitLinking => "Increase circuit isolation, reduce reuse",
            ThreatType::BandwidthFingerprinting => "Normalize packet sizes, constant-rate padding",
            ThreatType::IntersectionAttack => "Limit session duration, rotate identity",
            ThreatType::WebsiteFingerprinting => "Use cover traffic, randomize patterns",
            ThreatType::RelayEnumeration => "Use bridges, avoid predictable behavior",
        }
    }
}

/// Detected threat with details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedThreat {
    /// Type of threat
    pub threat_type: ThreatType,
    /// Risk level
    pub risk_level: RiskLevel,
    /// Description of the threat
    pub description: String,
    /// Recommended mitigation
    pub mitigation: String,
    /// When the threat was detected (not serialized)
    #[serde(skip, default = "Instant::now")]
    pub detected_at: Instant,
    /// Evidence supporting the detection
    pub evidence: Vec<String>,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
}

impl DetectedThreat {
    pub fn new(
        threat_type: ThreatType,
        risk_level: RiskLevel,
        description: String,
        confidence: f64,
    ) -> Self {
        Self {
            threat_type,
            risk_level,
            description,
            mitigation: threat_type.mitigation().to_string(),
            detected_at: Instant::now(),
            evidence: Vec::new(),
            confidence,
        }
    }

    pub fn with_evidence(mut self, evidence: Vec<String>) -> Self {
        self.evidence = evidence;
        self
    }
}

/// Configuration for anonymity monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymityMonitorConfig {
    /// Enable monitoring
    pub enabled: bool,
    /// Check interval
    pub check_interval: Duration,
    /// Maximum guard reuse before warning
    pub max_guard_reuse: u32,
    /// Circuit reuse threshold
    pub circuit_reuse_threshold: u32,
    /// Traffic volume window for correlation detection
    pub traffic_volume_window: Duration,
    /// Minimum relay diversity required
    pub min_relay_diversity: usize,
    /// Geographic diversity requirement (distinct countries)
    pub min_country_diversity: usize,
    /// Maximum acceptable timing variance (ms)
    pub max_timing_variance: u64,
    /// Enable automatic mitigation
    pub auto_mitigate: bool,
    /// Risk level threshold for alerts
    pub alert_threshold: RiskLevel,
}

impl Default for AnonymityMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_interval: Duration::from_secs(60),
            max_guard_reuse: 1000,
            circuit_reuse_threshold: 100,
            traffic_volume_window: Duration::from_secs(300), // 5 minutes
            min_relay_diversity: 10,
            min_country_diversity: 3,
            max_timing_variance: 50,
            auto_mitigate: true,
            alert_threshold: RiskLevel::Moderate,
        }
    }
}

/// Guard node usage statistics
#[derive(Debug, Clone, Default)]
struct GuardStats {
    /// Fingerprint -> usage count
    usage_count: HashMap<String, u32>,
    /// Last rotation time per guard
    last_rotation: HashMap<String, Instant>,
    /// Current guard set
    current_guards: HashSet<String>,
}

/// Circuit usage statistics
#[derive(Debug, Clone, Default)]
struct CircuitStats {
    /// Circuit ID -> usage count
    usage_count: HashMap<u64, u32>,
    /// Circuit ID -> creation time
    creation_time: HashMap<u64, Instant>,
    /// Recent circuit paths (for diversity checking)
    recent_paths: VecDeque<Vec<String>>,
}

/// Traffic pattern statistics
#[derive(Debug, Clone)]
struct TrafficStats {
    /// Sliding window of traffic volumes
    volume_samples: VecDeque<(Instant, u64)>,
    /// Sliding window of inter-packet times
    timing_samples: VecDeque<Duration>,
    /// Last traffic timestamp
    last_traffic: Option<Instant>,
    /// Bytes sent in current window
    bytes_sent: u64,
    /// Bytes received in current window
    bytes_received: u64,
}

impl Default for TrafficStats {
    fn default() -> Self {
        Self {
            volume_samples: VecDeque::with_capacity(1000),
            timing_samples: VecDeque::with_capacity(1000),
            last_traffic: None,
            bytes_sent: 0,
            bytes_received: 0,
        }
    }
}

/// Relay diversity information
#[derive(Debug, Clone, Default)]
struct RelayDiversity {
    /// Known relays by fingerprint
    known_relays: HashSet<String>,
    /// Relays by country code
    by_country: HashMap<String, HashSet<String>>,
    /// Relays by AS number
    by_as: HashMap<u32, HashSet<String>>,
    /// Relays by IP prefix (/16)
    by_prefix: HashMap<String, HashSet<String>>,
}

/// Anonymity assessment report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymityReport {
    /// Overall risk level
    pub overall_risk: RiskLevel,
    /// Overall risk score (0-100)
    pub risk_score: u8,
    /// Active threats
    pub threats: Vec<DetectedThreat>,
    /// Guard diversity score
    pub guard_diversity: f64,
    /// Circuit diversity score
    pub circuit_diversity: f64,
    /// Geographic diversity score
    pub geographic_diversity: f64,
    /// Traffic pattern score
    pub traffic_pattern_score: f64,
    /// Timing regularity score (lower is better)
    pub timing_regularity: f64,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Report generation time
    pub generated_at: SystemTime,
}

impl AnonymityReport {
    /// Create a new minimal-risk report
    pub fn minimal() -> Self {
        Self {
            overall_risk: RiskLevel::Minimal,
            risk_score: 10,
            threats: Vec::new(),
            guard_diversity: 1.0,
            circuit_diversity: 1.0,
            geographic_diversity: 1.0,
            traffic_pattern_score: 1.0,
            timing_regularity: 0.0,
            recommendations: Vec::new(),
            generated_at: SystemTime::now(),
        }
    }
}

/// Callback for threat notifications
pub type ThreatCallback = Arc<dyn Fn(DetectedThreat) + Send + Sync>;

/// Main anonymity monitoring manager
pub struct AnonymityMonitor {
    config: AnonymityMonitorConfig,
    guard_stats: Arc<RwLock<GuardStats>>,
    circuit_stats: Arc<RwLock<CircuitStats>>,
    traffic_stats: Arc<RwLock<TrafficStats>>,
    relay_diversity: Arc<RwLock<RelayDiversity>>,
    active_threats: Arc<RwLock<Vec<DetectedThreat>>>,
    threat_callbacks: Arc<RwLock<Vec<ThreatCallback>>>,
    last_check: Arc<RwLock<Option<Instant>>>,
    is_monitoring: Arc<std::sync::atomic::AtomicBool>,
}

impl AnonymityMonitor {
    /// Create a new anonymity monitor
    pub fn new(config: AnonymityMonitorConfig) -> Self {
        info!("🔍 Creating Anonymity Monitor");
        info!("   Check interval: {:?}", config.check_interval);
        info!("   Auto-mitigate: {}", config.auto_mitigate);
        info!("   Alert threshold: {}", config.alert_threshold.name());

        Self {
            config,
            guard_stats: Arc::new(RwLock::new(GuardStats::default())),
            circuit_stats: Arc::new(RwLock::new(CircuitStats::default())),
            traffic_stats: Arc::new(RwLock::new(TrafficStats::default())),
            relay_diversity: Arc::new(RwLock::new(RelayDiversity::default())),
            active_threats: Arc::new(RwLock::new(Vec::new())),
            threat_callbacks: Arc::new(RwLock::new(Vec::new())),
            last_check: Arc::new(RwLock::new(None)),
            is_monitoring: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Register a threat notification callback
    pub async fn on_threat(&self, callback: ThreatCallback) {
        let mut callbacks = self.threat_callbacks.write().await;
        callbacks.push(callback);
    }

    /// Record guard usage
    pub async fn record_guard_usage(&self, fingerprint: &str) {
        let mut stats = self.guard_stats.write().await;
        *stats.usage_count.entry(fingerprint.to_string()).or_insert(0) += 1;
        stats.current_guards.insert(fingerprint.to_string());
    }

    /// Record guard rotation
    pub async fn record_guard_rotation(&self, old_fingerprint: &str, new_fingerprint: &str) {
        let mut stats = self.guard_stats.write().await;
        stats.last_rotation.insert(old_fingerprint.to_string(), Instant::now());
        stats.current_guards.remove(old_fingerprint);
        stats.current_guards.insert(new_fingerprint.to_string());
        debug!("Guard rotated: {} -> {}", old_fingerprint, new_fingerprint);
    }

    /// Record circuit usage
    pub async fn record_circuit_usage(&self, circuit_id: u64, path: Vec<String>) {
        let mut stats = self.circuit_stats.write().await;
        *stats.usage_count.entry(circuit_id).or_insert(0) += 1;

        if !stats.creation_time.contains_key(&circuit_id) {
            stats.creation_time.insert(circuit_id, Instant::now());
        }

        // Keep last 100 paths for diversity analysis
        stats.recent_paths.push_back(path);
        if stats.recent_paths.len() > 100 {
            stats.recent_paths.pop_front();
        }
    }

    /// Record traffic event
    pub async fn record_traffic(&self, bytes_sent: u64, bytes_received: u64) {
        let mut stats = self.traffic_stats.write().await;
        let now = Instant::now();

        // Record timing sample
        if let Some(last) = stats.last_traffic {
            let interval = now.duration_since(last);
            stats.timing_samples.push_back(interval);
            if stats.timing_samples.len() > 1000 {
                stats.timing_samples.pop_front();
            }
        }
        stats.last_traffic = Some(now);

        // Record volume sample
        stats.volume_samples.push_back((now, bytes_sent + bytes_received));
        if stats.volume_samples.len() > 1000 {
            stats.volume_samples.pop_front();
        }

        stats.bytes_sent += bytes_sent;
        stats.bytes_received += bytes_received;
    }

    /// Add known relay
    pub async fn add_relay(&self, fingerprint: &str, country: Option<&str>, as_number: Option<u32>) {
        let mut diversity = self.relay_diversity.write().await;
        diversity.known_relays.insert(fingerprint.to_string());

        if let Some(country) = country {
            diversity.by_country
                .entry(country.to_string())
                .or_insert_with(HashSet::new)
                .insert(fingerprint.to_string());
        }

        if let Some(asn) = as_number {
            diversity.by_as
                .entry(asn)
                .or_insert_with(HashSet::new)
                .insert(fingerprint.to_string());
        }
    }

    /// Perform full anonymity assessment
    pub async fn assess(&self) -> AnonymityReport {
        let mut threats = Vec::new();
        let mut recommendations = Vec::new();

        // Check guard usage patterns
        let guard_diversity = self.assess_guard_diversity(&mut threats, &mut recommendations).await;

        // Check circuit diversity
        let circuit_diversity = self.assess_circuit_diversity(&mut threats, &mut recommendations).await;

        // Check geographic diversity
        let geographic_diversity = self.assess_geographic_diversity(&mut threats, &mut recommendations).await;

        // Check traffic patterns
        let (traffic_score, timing_regularity) = self.assess_traffic_patterns(&mut threats, &mut recommendations).await;

        // Calculate overall risk
        let (overall_risk, risk_score) = self.calculate_overall_risk(&threats);

        // Notify callbacks for high-risk threats
        if overall_risk >= self.config.alert_threshold {
            let callbacks = self.threat_callbacks.read().await;
            for threat in &threats {
                if threat.risk_level >= self.config.alert_threshold {
                    for callback in callbacks.iter() {
                        callback(threat.clone());
                    }
                }
            }
        }

        // Update active threats
        {
            let mut active = self.active_threats.write().await;
            *active = threats.clone();
        }

        // Update last check time
        {
            let mut last = self.last_check.write().await;
            *last = Some(Instant::now());
        }

        AnonymityReport {
            overall_risk,
            risk_score,
            threats,
            guard_diversity,
            circuit_diversity,
            geographic_diversity,
            traffic_pattern_score: traffic_score,
            timing_regularity,
            recommendations,
            generated_at: SystemTime::now(),
        }
    }

    /// Assess guard diversity
    async fn assess_guard_diversity(
        &self,
        threats: &mut Vec<DetectedThreat>,
        recommendations: &mut Vec<String>,
    ) -> f64 {
        let stats = self.guard_stats.read().await;

        // Check for excessive guard reuse
        let max_usage = stats.usage_count.values().max().copied().unwrap_or(0);
        if max_usage > self.config.max_guard_reuse {
            let threat = DetectedThreat::new(
                ThreatType::GuardDiscovery,
                RiskLevel::Moderate,
                format!("Guard reuse exceeds threshold: {} > {}", max_usage, self.config.max_guard_reuse),
                0.7,
            ).with_evidence(vec![
                format!("Max guard usage: {}", max_usage),
                format!("Threshold: {}", self.config.max_guard_reuse),
            ]);
            threats.push(threat);
            recommendations.push("Rotate guards more frequently".to_string());
        }

        // Check guard diversity
        let guard_count = stats.current_guards.len();
        if guard_count < 2 {
            let threat = DetectedThreat::new(
                ThreatType::GuardDiscovery,
                RiskLevel::High,
                format!("Insufficient guard diversity: {} guards", guard_count),
                0.9,
            );
            threats.push(threat);
            recommendations.push("Add more guard nodes to rotation".to_string());
        }

        // Calculate diversity score
        let ideal_guards = 4.0;
        let diversity = (guard_count as f64 / ideal_guards).min(1.0);
        diversity
    }

    /// Assess circuit diversity
    async fn assess_circuit_diversity(
        &self,
        threats: &mut Vec<DetectedThreat>,
        recommendations: &mut Vec<String>,
    ) -> f64 {
        let stats = self.circuit_stats.read().await;

        // Check for circuit reuse
        let max_reuse = stats.usage_count.values().max().copied().unwrap_or(0);
        if max_reuse > self.config.circuit_reuse_threshold {
            let threat = DetectedThreat::new(
                ThreatType::CircuitLinking,
                RiskLevel::Low,
                format!("Circuit reuse above threshold: {}", max_reuse),
                0.5,
            );
            threats.push(threat);
            recommendations.push("Reduce circuit reuse by creating new circuits more often".to_string());
        }

        // Check path diversity
        let unique_paths: HashSet<_> = stats.recent_paths.iter().map(|p| p.join(",")).collect();
        let path_diversity = if stats.recent_paths.is_empty() {
            1.0
        } else {
            unique_paths.len() as f64 / stats.recent_paths.len() as f64
        };

        if path_diversity < 0.3 {
            let threat = DetectedThreat::new(
                ThreatType::CircuitLinking,
                RiskLevel::Moderate,
                format!("Low circuit path diversity: {:.1}%", path_diversity * 100.0),
                0.6,
            );
            threats.push(threat);
            recommendations.push("Increase circuit path randomization".to_string());
        }

        path_diversity
    }

    /// Assess geographic diversity
    async fn assess_geographic_diversity(
        &self,
        threats: &mut Vec<DetectedThreat>,
        recommendations: &mut Vec<String>,
    ) -> f64 {
        let diversity = self.relay_diversity.read().await;

        let country_count = diversity.by_country.len();
        if country_count < self.config.min_country_diversity {
            let threat = DetectedThreat::new(
                ThreatType::GeographicCorrelation,
                RiskLevel::Low,
                format!("Limited geographic diversity: {} countries", country_count),
                0.4,
            );
            threats.push(threat);
            recommendations.push("Use relays from more diverse geographic locations".to_string());
        }

        // Check AS diversity
        let as_count = diversity.by_as.len();
        if as_count < 5 {
            let threat = DetectedThreat::new(
                ThreatType::SybilAttack,
                RiskLevel::Moderate,
                format!("Limited AS diversity: {} autonomous systems", as_count),
                0.5,
            );
            threats.push(threat);
            recommendations.push("Distribute relays across more autonomous systems".to_string());
        }

        // Calculate diversity score
        let country_score = (country_count as f64 / self.config.min_country_diversity as f64).min(1.0);
        let as_score = (as_count as f64 / 10.0).min(1.0);
        (country_score + as_score) / 2.0
    }

    /// Assess traffic patterns
    async fn assess_traffic_patterns(
        &self,
        threats: &mut Vec<DetectedThreat>,
        recommendations: &mut Vec<String>,
    ) -> (f64, f64) {
        let stats = self.traffic_stats.read().await;

        // Calculate timing regularity
        let timing_regularity = self.calculate_timing_regularity(&stats.timing_samples);

        if timing_regularity > 0.8 {
            let threat = DetectedThreat::new(
                ThreatType::TimingAnalysis,
                RiskLevel::Moderate,
                format!("High timing regularity detected: {:.1}%", timing_regularity * 100.0),
                0.6,
            );
            threats.push(threat);
            recommendations.push("Add timing jitter to packet transmission".to_string());
        }

        // Calculate volume variance
        let volume_variance = self.calculate_volume_variance(&stats.volume_samples);

        if volume_variance < 0.1 {
            let threat = DetectedThreat::new(
                ThreatType::BandwidthFingerprinting,
                RiskLevel::Low,
                "Traffic volume is highly predictable".to_string(),
                0.4,
            );
            threats.push(threat);
            recommendations.push("Add cover traffic and volume padding".to_string());
        }

        // Traffic pattern score (higher variance is better)
        let traffic_score = (volume_variance * 2.0).min(1.0);

        (traffic_score, timing_regularity)
    }

    /// Calculate timing regularity (0.0 = random, 1.0 = perfectly regular)
    fn calculate_timing_regularity(&self, samples: &VecDeque<Duration>) -> f64 {
        if samples.len() < 10 {
            return 0.0; // Not enough samples
        }

        let intervals: Vec<f64> = samples.iter().map(|d| d.as_micros() as f64).collect();
        let mean: f64 = intervals.iter().sum::<f64>() / intervals.len() as f64;

        if mean == 0.0 {
            return 0.0;
        }

        let variance: f64 = intervals.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / intervals.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = std_dev / mean;

        // Lower CV means more regular (predictable) timing
        // Convert to 0-1 scale where 1 = highly regular
        (1.0 - coefficient_of_variation.min(1.0)).max(0.0)
    }

    /// Calculate volume variance
    fn calculate_volume_variance(&self, samples: &VecDeque<(Instant, u64)>) -> f64 {
        if samples.len() < 10 {
            return 0.5; // Default moderate variance
        }

        let volumes: Vec<f64> = samples.iter().map(|(_, v)| *v as f64).collect();
        let mean: f64 = volumes.iter().sum::<f64>() / volumes.len() as f64;

        if mean == 0.0 {
            return 0.0;
        }

        let variance: f64 = volumes.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / volumes.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = std_dev / mean;

        coefficient_of_variation.min(1.0)
    }

    /// Calculate overall risk from threats
    fn calculate_overall_risk(&self, threats: &[DetectedThreat]) -> (RiskLevel, u8) {
        if threats.is_empty() {
            return (RiskLevel::Minimal, 10);
        }

        // Find highest risk level
        let max_risk = threats.iter()
            .map(|t| t.risk_level)
            .max()
            .unwrap_or(RiskLevel::Minimal);

        // Calculate weighted score
        let total_score: f64 = threats.iter()
            .map(|t| t.risk_level.score() as f64 * t.confidence)
            .sum();

        let avg_score = (total_score / threats.len() as f64) as u8;

        // Bump up if multiple high-risk threats
        let high_risk_count = threats.iter()
            .filter(|t| t.risk_level >= RiskLevel::High)
            .count();

        let final_risk = if high_risk_count >= 3 {
            RiskLevel::Critical
        } else if high_risk_count >= 2 {
            RiskLevel::High
        } else {
            max_risk
        };

        (final_risk, avg_score.max(final_risk.score()))
    }

    /// Get current active threats
    pub async fn get_active_threats(&self) -> Vec<DetectedThreat> {
        self.active_threats.read().await.clone()
    }

    /// Get quick risk summary
    pub async fn get_risk_summary(&self) -> (RiskLevel, u8) {
        let threats = self.active_threats.read().await;
        self.calculate_overall_risk(&threats)
    }

    /// Start background monitoring
    pub async fn start_monitoring(&self) {
        use std::sync::atomic::Ordering;

        if self.is_monitoring.swap(true, Ordering::SeqCst) {
            warn!("Anonymity monitoring already running");
            return;
        }

        info!("🔍 Starting anonymity monitoring");

        let config = self.config.clone();
        let monitor = AnonymityMonitor {
            config: self.config.clone(),
            guard_stats: Arc::clone(&self.guard_stats),
            circuit_stats: Arc::clone(&self.circuit_stats),
            traffic_stats: Arc::clone(&self.traffic_stats),
            relay_diversity: Arc::clone(&self.relay_diversity),
            active_threats: Arc::clone(&self.active_threats),
            threat_callbacks: Arc::clone(&self.threat_callbacks),
            last_check: Arc::clone(&self.last_check),
            is_monitoring: Arc::clone(&self.is_monitoring),
        };

        tokio::spawn(async move {
            loop {
                if !monitor.is_monitoring.load(Ordering::SeqCst) {
                    break;
                }

                // Perform assessment
                let report = monitor.assess().await;

                if report.overall_risk >= config.alert_threshold {
                    warn!(
                        "⚠️ Anonymity risk elevated: {} (score: {})",
                        report.overall_risk.name(),
                        report.risk_score
                    );
                } else {
                    debug!(
                        "Anonymity check: {} (score: {})",
                        report.overall_risk.name(),
                        report.risk_score
                    );
                }

                tokio::time::sleep(config.check_interval).await;
            }

            info!("🔍 Anonymity monitoring stopped");
        });
    }

    /// Stop background monitoring
    pub fn stop_monitoring(&self) {
        use std::sync::atomic::Ordering;
        self.is_monitoring.store(false, Ordering::SeqCst);
    }
}

/// Anonymity-aware circuit selection
pub struct AnonymityAwareSelector {
    monitor: Arc<AnonymityMonitor>,
    /// Maximum risk level for automatic operations
    max_auto_risk: RiskLevel,
}

impl AnonymityAwareSelector {
    pub fn new(monitor: Arc<AnonymityMonitor>) -> Self {
        Self {
            monitor,
            max_auto_risk: RiskLevel::Moderate,
        }
    }

    /// Check if it's safe to proceed with an operation
    pub async fn is_safe_to_proceed(&self) -> bool {
        let (risk, _) = self.monitor.get_risk_summary().await;
        risk <= self.max_auto_risk
    }

    /// Get recommended action based on current risk
    pub async fn get_recommended_action(&self) -> AnonymityAction {
        let (risk, _) = self.monitor.get_risk_summary().await;

        match risk {
            RiskLevel::Minimal | RiskLevel::Low => AnonymityAction::Proceed,
            RiskLevel::Moderate => AnonymityAction::ProceedWithCaution,
            RiskLevel::High => AnonymityAction::RotateCircuits,
            RiskLevel::Critical => AnonymityAction::PauseOperations,
        }
    }
}

/// Recommended action based on anonymity assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnonymityAction {
    /// Safe to proceed normally
    Proceed,
    /// Proceed but increase monitoring
    ProceedWithCaution,
    /// Rotate circuits before proceeding
    RotateCircuits,
    /// Pause all operations until risk decreases
    PauseOperations,
}

impl AnonymityAction {
    pub fn name(&self) -> &'static str {
        match self {
            AnonymityAction::Proceed => "Proceed",
            AnonymityAction::ProceedWithCaution => "Proceed with Caution",
            AnonymityAction::RotateCircuits => "Rotate Circuits",
            AnonymityAction::PauseOperations => "Pause Operations",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_anonymity_monitor_creation() {
        let config = AnonymityMonitorConfig::default();
        let monitor = AnonymityMonitor::new(config);

        let report = monitor.assess().await;
        // A new monitor with no diversity data will have elevated risk
        // due to zero geographic/AS diversity - this is expected behavior
        assert!(report.risk_score > 0);
        assert!(report.recommendations.len() > 0);
    }

    #[tokio::test]
    async fn test_guard_usage_tracking() {
        let config = AnonymityMonitorConfig::default();
        let monitor = AnonymityMonitor::new(config);

        // Record guard usage
        for _ in 0..100 {
            monitor.record_guard_usage("FINGERPRINT1").await;
        }

        let report = monitor.assess().await;
        assert!(report.guard_diversity <= 1.0);
    }

    #[tokio::test]
    async fn test_threat_detection() {
        let mut config = AnonymityMonitorConfig::default();
        config.max_guard_reuse = 10; // Low threshold for testing
        let monitor = AnonymityMonitor::new(config);

        // Create excessive guard reuse
        for _ in 0..100 {
            monitor.record_guard_usage("FINGERPRINT1").await;
        }

        let report = monitor.assess().await;
        assert!(!report.threats.is_empty());
    }

    #[test]
    fn test_risk_level_ordering() {
        assert!(RiskLevel::Minimal < RiskLevel::Low);
        assert!(RiskLevel::Low < RiskLevel::Moderate);
        assert!(RiskLevel::Moderate < RiskLevel::High);
        assert!(RiskLevel::High < RiskLevel::Critical);
    }
}
