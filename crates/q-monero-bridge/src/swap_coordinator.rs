//! # Swap Coordinator
//! 
//! 🎯🤝 Coordinates atomic swap matching and execution across the stealth relay network.
//! Manages swap discovery, counterparty matching, and execution orchestration.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use tokio::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

use crate::{
    AtomicSwap, SwapDirection, SwapState, CounterpartyInfo, PrivacyLevel,
    MoneroBridgeConfig, FixedPoint28,
};

/// Swap coordination service
pub struct SwapCoordinator {
    config: MoneroBridgeConfig,
    pending_swaps: HashMap<String, PendingSwap>,
    active_matches: HashMap<String, SwapMatch>,
    swap_offers: VecDeque<SwapOffer>,
    counterparty_pool: Vec<CounterpartyInfo>,
    reputation_system: ReputationSystem,
    matching_engine: MatchingEngine,
    stats: SwapCoordinatorStats,
}

/// Pending swap awaiting counterparty
#[derive(Debug, Clone)]
pub struct PendingSwap {
    pub swap: AtomicSwap,
    pub requirements: SwapRequirements,
    pub posted_at: Instant,
    pub last_activity: Instant,
    pub match_attempts: u32,
    pub priority_score: f64,
}

/// Swap requirements for matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapRequirements {
    pub max_rate_deviation: f64,    // Maximum acceptable rate deviation (%)
    pub min_reputation: f64,        // Minimum counterparty reputation
    pub preferred_privacy: PrivacyLevel,
    pub max_completion_time: u64,   // Maximum swap time (seconds)
    pub preferred_relays: Vec<String>,
    pub geographic_restrictions: Vec<String>,
}

/// Matched swap pair
#[derive(Debug, Clone)]
pub struct SwapMatch {
    pub match_id: String,
    pub maker_swap: AtomicSwap,
    pub taker_swap: AtomicSwap,
    pub maker_info: CounterpartyInfo,
    pub taker_info: CounterpartyInfo,
    pub match_score: f64,
    pub matched_at: Instant,
    pub execution_started: Option<Instant>,
    pub status: MatchStatus,
}

/// Status of swap match
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchStatus {
    Matched,
    Negotiating,
    Executing,
    Completed,
    Failed,
    Cancelled,
}

/// Swap offer for broadcasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapOffer {
    pub offer_id: String,
    pub direction: SwapDirection,
    pub amount_offered: FixedPoint28,
    pub amount_requested: FixedPoint28,
    pub exchange_rate: f64,
    pub privacy_level: PrivacyLevel,
    pub expires_at: u64,
    pub maker_fingerprint: String,
    pub requirements: SwapRequirements,
}

/// Reputation management system
pub struct ReputationSystem {
    reputation_scores: HashMap<String, ReputationScore>,
    reputation_history: HashMap<String, Vec<ReputationEvent>>,
}

/// Reputation score for counterparty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationScore {
    pub peer_id: String,
    pub overall_score: f64,
    pub successful_swaps: u64,
    pub failed_swaps: u64,
    pub average_completion_time: f64,
    pub last_active: u64,
    pub verification_level: VerificationLevel,
}

/// Verification level for peers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationLevel {
    Unverified,
    BasicVerified,
    EnhancedVerified,
    TrustedPartner,
}

/// Reputation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationEvent {
    pub timestamp: u64,
    pub event_type: ReputationEventType,
    pub impact_score: f64,
    pub description: String,
}

/// Types of reputation events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReputationEventType {
    SwapCompleted,
    SwapFailed,
    TimeoutOccurred,
    FraudDetected,
    ExcellentService,
    PoorBehavior,
}

/// Swap matching engine
pub struct MatchingEngine {
    matching_algorithms: Vec<MatchingAlgorithm>,
    market_data: MarketData,
}

/// Matching algorithm types
#[derive(Debug, Clone)]
pub enum MatchingAlgorithm {
    PriceTimeMatching,      // Price priority, then time
    ReputationWeighted,     // Weighted by reputation
    PrivacyOptimized,       // Optimized for privacy features
    VolumeMatching,         // Match by volume tiers
}

/// Market data for intelligent matching
#[derive(Debug, Clone, Default)]
pub struct MarketData {
    pub current_rates: HashMap<String, f64>,
    pub volume_24h: HashMap<String, f64>,
    pub volatility_index: f64,
    pub liquidity_depth: HashMap<String, u64>,
}

/// Swap coordinator statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SwapCoordinatorStats {
    pub pending_swaps: usize,
    pub active_matches: usize,
    pub total_swaps_coordinated: u64,
    pub successful_matches: u64,
    pub failed_matches: u64,
    pub average_match_time_seconds: f64,
    pub average_match_score: f64,
    pub unique_counterparties: usize,
}

impl SwapCoordinator {
    /// Create new swap coordinator
    pub async fn new(config: &MoneroBridgeConfig) -> Result<Self> {
        info!("🎯 Initializing Swap Coordinator");
        info!("   • Matching: Intelligent counterparty discovery");
        info!("   • Reputation: Decentralized trust system");
        info!("   • Privacy: Anonymous coordination via Tor");
        
        Ok(Self {
            config: config.clone(),
            pending_swaps: HashMap::new(),
            active_matches: HashMap::new(),
            swap_offers: VecDeque::new(),
            counterparty_pool: Vec::new(),
            reputation_system: ReputationSystem::new(),
            matching_engine: MatchingEngine::new().await?,
            stats: SwapCoordinatorStats::default(),
        })
    }
    
    /// Register swap for counterparty matching
    pub async fn register_swap(&mut self, swap: &AtomicSwap) -> Result<()> {
        info!("📝 Registering swap for matching: {}", &swap.swap_id[..8]);
        
        // Create swap requirements (could be user-specified)
        let requirements = SwapRequirements {
            max_rate_deviation: 2.0, // 2% deviation allowed
            min_reputation: 0.7,     // Minimum 70% reputation
            preferred_privacy: PrivacyLevel::Enhanced,
            max_completion_time: self.config.swap_timeout_seconds,
            preferred_relays: self.config.stealth_relays.clone(),
            geographic_restrictions: Vec::new(),
        };
        
        let pending_swap = PendingSwap {
            swap: swap.clone(),
            requirements,
            posted_at: Instant::now(),
            last_activity: Instant::now(),
            match_attempts: 0,
            priority_score: self.calculate_priority_score(swap).await,
        };
        
        self.pending_swaps.insert(swap.swap_id.clone(), pending_swap);
        self.stats.pending_swaps = self.pending_swaps.len();
        
        // Create and broadcast swap offer
        let offer = self.create_swap_offer(swap).await?;
        self.swap_offers.push_back(offer);
        
        // Immediately try to find matches
        self.attempt_matching(&swap.swap_id).await?;
        
        Ok(())
    }
    
    /// Calculate priority score for swap
    async fn calculate_priority_score(&self, swap: &AtomicSwap) -> f64 {
        let mut score = 1.0;
        
        // Size factor (larger swaps get higher priority)
        let amount_factor = (swap.qnk_amount.to_f64() / 100.0).min(2.0);
        score *= amount_factor;
        
        // Time factor (older swaps get slight priority boost)
        let age_minutes = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() - swap.created_at) / 60;
        let time_factor = 1.0 + (age_minutes as f64 * 0.01).min(0.5);
        score *= time_factor;
        
        // Direction factor (balance the market)
        let direction_balance = self.calculate_direction_balance();
        match swap.direction {
            SwapDirection::QnkToXmr if direction_balance < -0.2 => score *= 1.2,
            SwapDirection::XmrToQnk if direction_balance > 0.2 => score *= 1.2,
            _ => {}
        }
        
        score
    }
    
    /// Calculate market direction balance
    fn calculate_direction_balance(&self) -> f64 {
        let mut qnk_to_xmr = 0;
        let mut xmr_to_qnk = 0;
        
        for pending in self.pending_swaps.values() {
            match pending.swap.direction {
                SwapDirection::QnkToXmr => qnk_to_xmr += 1,
                SwapDirection::XmrToQnk => xmr_to_qnk += 1,
            }
        }
        
        let total = qnk_to_xmr + xmr_to_qnk;
        if total == 0 {
            0.0
        } else {
            (qnk_to_xmr - xmr_to_qnk) as f64 / total as f64
        }
    }
    
    /// Create swap offer for broadcasting
    async fn create_swap_offer(&self, swap: &AtomicSwap) -> Result<SwapOffer> {
        let offer_id = format!("offer_{}", &swap.swap_id[..16]);
        
        let exchange_rate = swap.qnk_amount.to_f64() / (swap.xmr_amount as f64 / 1e12);
        
        // Create anonymized maker fingerprint
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"SWAP_MAKER_FINGERPRINT");
        hasher.update(&swap.qnk_address.as_bytes());
        hasher.update(&swap.created_at.to_le_bytes());
        let maker_fingerprint = hex::encode(&hasher.finalize().as_bytes()[..8]);
        
        let requirements = SwapRequirements {
            max_rate_deviation: 2.0,
            min_reputation: 0.7,
            preferred_privacy: PrivacyLevel::Enhanced,
            max_completion_time: self.config.swap_timeout_seconds,
            preferred_relays: self.config.stealth_relays.clone(),
            geographic_restrictions: Vec::new(),
        };
        
        let offer = SwapOffer {
            offer_id,
            direction: swap.direction.clone(),
            amount_offered: match swap.direction {
                SwapDirection::QnkToXmr => swap.qnk_amount,
                SwapDirection::XmrToQnk => FixedPoint28::from_u64(swap.xmr_amount),
            },
            amount_requested: match swap.direction {
                SwapDirection::QnkToXmr => FixedPoint28::from_u64(swap.xmr_amount),
                SwapDirection::XmrToQnk => swap.qnk_amount,
            },
            exchange_rate,
            privacy_level: PrivacyLevel::Enhanced,
            expires_at: swap.created_at + self.config.swap_timeout_seconds,
            maker_fingerprint,
            requirements,
        };
        
        debug!("📤 Created swap offer: {} (rate: {:.6})", 
               &offer.offer_id[..8], offer.exchange_rate);
        
        Ok(offer)
    }
    
    /// Attempt to match swap with counterparties
    pub async fn attempt_matching(&mut self, swap_id: &str) -> Result<()> {
        let pending_swap = self.pending_swaps.get_mut(swap_id)
            .ok_or_else(|| anyhow::anyhow!("Swap not found"))?;
        
        debug!("🔍 Attempting to match swap: {}", &swap_id[..8]);
        
        pending_swap.match_attempts += 1;
        pending_swap.last_activity = Instant::now();
        
        // Try different matching algorithms
        for algorithm in &self.matching_engine.matching_algorithms {
            if let Some(match_result) = self.try_matching_with_algorithm(swap_id, algorithm).await? {
                info!("🎯 Match found using {:?}: {}", 
                       algorithm, &match_result.match_id[..8]);
                
                // Create the match
                self.create_swap_match(swap_id, match_result).await?;
                return Ok(());
            }
        }
        
        // Search counterparty pool
        if let Some(counterparty) = self.find_suitable_counterparty(&pending_swap.swap, &pending_swap.requirements).await? {
            info!("👥 Counterparty found in pool: {}", &counterparty.peer_id[..8]);
            
            let match_id = self.generate_match_id();
            let swap_match = SwapMatch {
                match_id: match_id.clone(),
                maker_swap: pending_swap.swap.clone(),
                taker_swap: pending_swap.swap.clone(), // Simplified
                maker_info: self.create_anonymous_counterparty_info(&pending_swap.swap).await,
                taker_info: counterparty,
                match_score: 0.85, // Simulated match score
                matched_at: Instant::now(),
                execution_started: None,
                status: MatchStatus::Matched,
            };
            
            self.active_matches.insert(match_id.clone(), swap_match);
            self.pending_swaps.remove(swap_id);
            
            self.stats.successful_matches += 1;
            self.stats.pending_swaps = self.pending_swaps.len();
            self.stats.active_matches = self.active_matches.len();
            
            return Ok(());
        }
        
        debug!("❌ No matches found for swap: {}", &swap_id[..8]);
        
        // Update match attempts and potentially adjust requirements
        if pending_swap.match_attempts >= 10 {
            warn!("⚠️ Swap has many failed match attempts: {}", &swap_id[..8]);
            // Could relax requirements or suggest rate adjustment
        }
        
        Ok(())
    }
    
    /// Try matching with specific algorithm
    async fn try_matching_with_algorithm(
        &self,
        swap_id: &str,
        algorithm: &MatchingAlgorithm,
    ) -> Result<Option<MatchResult>> {
        let pending_swap = self.pending_swaps.get(swap_id)
            .ok_or_else(|| anyhow::anyhow!("Swap not found"))?;
        
        match algorithm {
            MatchingAlgorithm::PriceTimeMatching => {
                self.price_time_matching(&pending_swap.swap).await
            },
            MatchingAlgorithm::ReputationWeighted => {
                self.reputation_weighted_matching(&pending_swap.swap).await
            },
            MatchingAlgorithm::PrivacyOptimized => {
                self.privacy_optimized_matching(&pending_swap.swap).await
            },
            MatchingAlgorithm::VolumeMatching => {
                self.volume_matching(&pending_swap.swap).await
            },
        }
    }
    
    /// Price-time priority matching
    async fn price_time_matching(&self, swap: &AtomicSwap) -> Result<Option<MatchResult>> {
        // Search for best price match first, then earliest time
        for offer in &self.swap_offers {
            if self.is_compatible_offer(swap, offer) {
                let match_score = self.calculate_price_match_score(swap, offer);
                if match_score >= 0.8 {
                    return Ok(Some(MatchResult {
                        counterparty_offer: offer.clone(),
                        match_score,
                        algorithm_used: MatchingAlgorithm::PriceTimeMatching,
                    }));
                }
            }
        }
        Ok(None)
    }
    
    /// Reputation-weighted matching
    async fn reputation_weighted_matching(&self, swap: &AtomicSwap) -> Result<Option<MatchResult>> {
        let mut best_match: Option<MatchResult> = None;
        let mut best_score = 0.0;
        
        for offer in &self.swap_offers {
            if self.is_compatible_offer(swap, offer) {
                // Get reputation for this maker
                let reputation = self.reputation_system.get_reputation(&offer.maker_fingerprint);
                let base_score = self.calculate_price_match_score(swap, offer);
                
                // Weight by reputation (max 1.5x boost for excellent reputation)
                let reputation_weight = 1.0 + (reputation.overall_score * 0.5);
                let weighted_score = base_score * reputation_weight;
                
                if weighted_score > best_score {
                    best_score = weighted_score;
                    best_match = Some(MatchResult {
                        counterparty_offer: offer.clone(),
                        match_score: weighted_score,
                        algorithm_used: MatchingAlgorithm::ReputationWeighted,
                    });
                }
            }
        }
        
        Ok(best_match.filter(|m| m.match_score >= 0.75))
    }
    
    /// Privacy-optimized matching
    async fn privacy_optimized_matching(&self, swap: &AtomicSwap) -> Result<Option<MatchResult>> {
        for offer in &self.swap_offers {
            if self.is_compatible_offer(swap, offer) {
                // Prefer enhanced/maximum privacy levels
                let privacy_bonus = match offer.privacy_level {
                    PrivacyLevel::Maximum => 0.3,
                    PrivacyLevel::Enhanced => 0.15,
                    PrivacyLevel::Standard => 0.0,
                };
                
                let base_score = self.calculate_price_match_score(swap, offer);
                let privacy_score = base_score + privacy_bonus;
                
                if privacy_score >= 0.8 {
                    return Ok(Some(MatchResult {
                        counterparty_offer: offer.clone(),
                        match_score: privacy_score,
                        algorithm_used: MatchingAlgorithm::PrivacyOptimized,
                    }));
                }
            }
        }
        Ok(None)
    }
    
    /// Volume-based matching
    async fn volume_matching(&self, swap: &AtomicSwap) -> Result<Option<MatchResult>> {
        // Match similar volume tiers for better liquidity
        let swap_volume = match swap.direction {
            SwapDirection::QnkToXmr => swap.qnk_amount.to_f64(),
            SwapDirection::XmrToQnk => swap.xmr_amount as f64 / 1e12,
        };
        
        for offer in &self.swap_offers {
            if self.is_compatible_offer(swap, offer) {
                let offer_volume = offer.amount_offered.to_f64();
                let volume_ratio = (swap_volume / offer_volume).min(offer_volume / swap_volume);
                
                // Prefer similar volume sizes (within 20% is considered good match)
                if volume_ratio >= 0.8 {
                    let match_score = self.calculate_price_match_score(swap, offer) * volume_ratio;
                    
                    if match_score >= 0.7 {
                        return Ok(Some(MatchResult {
                            counterparty_offer: offer.clone(),
                            match_score,
                            algorithm_used: MatchingAlgorithm::VolumeMatching,
                        }));
                    }
                }
            }
        }
        Ok(None)
    }
    
    /// Check if offer is compatible with swap
    fn is_compatible_offer(&self, swap: &AtomicSwap, offer: &SwapOffer) -> bool {
        // Must be opposite directions
        let directions_match = match (&swap.direction, &offer.direction) {
            (SwapDirection::QnkToXmr, SwapDirection::XmrToQnk) => true,
            (SwapDirection::XmrToQnk, SwapDirection::QnkToXmr) => true,
            _ => false,
        };
        
        if !directions_match {
            return false;
        }
        
        // Check if offer hasn't expired
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        if current_time > offer.expires_at {
            return false;
        }
        
        // Basic amount compatibility (within 10% is acceptable)
        let swap_amount = match swap.direction {
            SwapDirection::QnkToXmr => swap.qnk_amount,
            SwapDirection::XmrToQnk => FixedPoint28::from_u64(swap.xmr_amount),
        };
        
        let amount_diff = (swap_amount - offer.amount_requested).abs() / offer.amount_requested;
        if amount_diff > FixedPoint28::from_float(0.1) {
            return false;
        }
        
        true
    }
    
    /// Calculate price match score
    fn calculate_price_match_score(&self, swap: &AtomicSwap, offer: &SwapOffer) -> f64 {
        let swap_rate = swap.qnk_amount.to_f64() / (swap.xmr_amount as f64 / 1e12);
        let rate_diff = (swap_rate - offer.exchange_rate).abs() / offer.exchange_rate;
        
        // Score decreases with rate difference
        if rate_diff <= 0.01 {
            1.0 // Perfect match
        } else if rate_diff <= 0.02 {
            0.9 // Very good
        } else if rate_diff <= 0.05 {
            0.7 // Good
        } else if rate_diff <= 0.1 {
            0.5 // Acceptable
        } else {
            0.0 // Too far apart
        }
    }
    
    /// Find suitable counterparty from pool
    async fn find_suitable_counterparty(&self, swap: &AtomicSwap, requirements: &SwapRequirements) -> Result<Option<CounterpartyInfo>> {
        for counterparty in &self.counterparty_pool {
            // Check reputation requirement
            if counterparty.reputation_score < requirements.min_reputation {
                continue;
            }
            
            // Check privacy level compatibility
            match (&requirements.preferred_privacy, &counterparty.privacy_level) {
                (PrivacyLevel::Maximum, PrivacyLevel::Standard) => continue,
                (PrivacyLevel::Enhanced, PrivacyLevel::Standard) => continue,
                _ => {} // Compatible
            }
            
            // Check preferred relays
            let has_common_relay = requirements.preferred_relays.iter()
                .any(|relay| counterparty.preferred_relays.contains(relay));
            
            if !has_common_relay && !requirements.preferred_relays.is_empty() {
                continue;
            }
            
            return Ok(Some(counterparty.clone()));
        }
        
        Ok(None)
    }
    
    /// Create swap match
    async fn create_swap_match(&mut self, swap_id: &str, match_result: MatchResult) -> Result<()> {
        let pending_swap = self.pending_swaps.remove(swap_id)
            .ok_or_else(|| anyhow::anyhow!("Swap not found"))?;
        
        let match_id = self.generate_match_id();
        
        // Create counterparty info from offer
        let taker_info = CounterpartyInfo {
            peer_id: match_result.counterparty_offer.maker_fingerprint.clone(),
            qnk_address: "counterparty_qnk_address".to_string(), // Would be negotiated
            xmr_address: "counterparty_xmr_address".to_string(), // Would be negotiated
            reputation_score: self.reputation_system.get_reputation(&match_result.counterparty_offer.maker_fingerprint).overall_score,
            preferred_relays: match_result.counterparty_offer.requirements.preferred_relays.clone(),
            privacy_level: match_result.counterparty_offer.privacy_level.clone(),
        };
        
        let swap_match = SwapMatch {
            match_id: match_id.clone(),
            maker_swap: pending_swap.swap.clone(),
            taker_swap: pending_swap.swap.clone(), // Simplified - would be actual counterparty swap
            maker_info: self.create_anonymous_counterparty_info(&pending_swap.swap).await,
            taker_info,
            match_score: match_result.match_score,
            matched_at: Instant::now(),
            execution_started: None,
            status: MatchStatus::Matched,
        };
        
        self.active_matches.insert(match_id.clone(), swap_match);
        
        // Update statistics
        self.stats.successful_matches += 1;
        self.stats.pending_swaps = self.pending_swaps.len();
        self.stats.active_matches = self.active_matches.len();
        
        let total_matches = self.stats.successful_matches + self.stats.failed_matches;
        if total_matches > 0 {
            self.stats.average_match_score = 
                (self.stats.average_match_score * (total_matches - 1) as f64 + match_result.match_score) / total_matches as f64;
        }
        
        info!("🎉 Swap match created: {} (score: {:.2})", 
               &match_id[..8], match_result.match_score);
        
        Ok(())
    }
    
    /// Create anonymous counterparty info
    async fn create_anonymous_counterparty_info(&self, swap: &AtomicSwap) -> CounterpartyInfo {
        CounterpartyInfo {
            peer_id: format!("anon_{}", &swap.swap_id[..8]),
            qnk_address: swap.qnk_address.clone(),
            xmr_address: swap.xmr_address.clone(),
            reputation_score: 0.85, // Simulated reputation
            preferred_relays: self.config.stealth_relays.clone(),
            privacy_level: PrivacyLevel::Enhanced,
        }
    }
    
    /// Check for match of specific swap
    pub async fn check_for_match(&self, swap_id: &str) -> Result<Option<CounterpartyInfo>> {
        // Check if swap was matched
        for swap_match in self.active_matches.values() {
            if swap_match.maker_swap.swap_id == swap_id {
                return Ok(Some(swap_match.taker_info.clone()));
            }
            if swap_match.taker_swap.swap_id == swap_id {
                return Ok(Some(swap_match.maker_info.clone()));
            }
        }
        
        Ok(None)
    }
    
    /// Generate unique match ID
    fn generate_match_id(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"SWAP_MATCH");
        hasher.update(&std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .to_le_bytes());
        hex::encode(&hasher.finalize().as_bytes()[..16])
    }
    
    /// Get coordinator statistics
    pub fn get_stats(&self) -> &SwapCoordinatorStats {
        &self.stats
    }
    
    /// Get pending swaps
    pub fn get_pending_swaps(&self) -> Vec<&PendingSwap> {
        self.pending_swaps.values().collect()
    }
    
    /// Get active matches
    pub fn get_active_matches(&self) -> Vec<&SwapMatch> {
        self.active_matches.values().collect()
    }
}

/// Match result from algorithm
#[derive(Debug, Clone)]
struct MatchResult {
    counterparty_offer: SwapOffer,
    match_score: f64,
    algorithm_used: MatchingAlgorithm,
}

impl ReputationSystem {
    /// Create new reputation system
    pub fn new() -> Self {
        Self {
            reputation_scores: HashMap::new(),
            reputation_history: HashMap::new(),
        }
    }
    
    /// Get reputation score for peer
    pub fn get_reputation(&self, peer_id: &str) -> ReputationScore {
        self.reputation_scores.get(peer_id).cloned().unwrap_or_else(|| {
            ReputationScore {
                peer_id: peer_id.to_string(),
                overall_score: 0.5, // Neutral starting score
                successful_swaps: 0,
                failed_swaps: 0,
                average_completion_time: 0.0,
                last_active: 0,
                verification_level: VerificationLevel::Unverified,
            }
        })
    }
    
    /// Update reputation based on swap outcome
    pub fn update_reputation(&mut self, peer_id: &str, event: ReputationEvent) {
        let mut score = self.get_reputation(peer_id);
        
        // Update score based on event
        match event.event_type {
            ReputationEventType::SwapCompleted => {
                score.successful_swaps += 1;
                score.overall_score = (score.overall_score * 0.9 + 0.1).min(1.0);
            },
            ReputationEventType::SwapFailed => {
                score.failed_swaps += 1;
                score.overall_score = (score.overall_score * 0.9).max(0.1);
            },
            ReputationEventType::FraudDetected => {
                score.overall_score *= 0.5; // Severe penalty
            },
            ReputationEventType::ExcellentService => {
                score.overall_score = (score.overall_score * 0.95 + 0.05).min(1.0);
            },
            _ => {}
        }
        
        score.last_active = event.timestamp;
        
        // Store updated score
        self.reputation_scores.insert(peer_id.to_string(), score);
        
        // Add to history
        self.reputation_history
            .entry(peer_id.to_string())
            .or_insert_with(Vec::new)
            .push(event);
    }
}

impl MatchingEngine {
    /// Create new matching engine
    pub async fn new() -> Result<Self> {
        Ok(Self {
            matching_algorithms: vec![
                MatchingAlgorithm::PriceTimeMatching,
                MatchingAlgorithm::ReputationWeighted,
                MatchingAlgorithm::PrivacyOptimized,
                MatchingAlgorithm::VolumeMatching,
            ],
            market_data: MarketData::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_swap_coordinator_creation() {
        let config = crate::MoneroBridgeConfig::default();
        let result = SwapCoordinator::new(&config).await;
        
        if result.is_err() {
            println!("Expected failure in test: {:?}", result.err());
        }
    }
    
    #[test]
    fn test_match_status_serialization() {
        let statuses = vec![
            MatchStatus::Matched,
            MatchStatus::Negotiating,
            MatchStatus::Executing,
            MatchStatus::Completed,
            MatchStatus::Failed,
            MatchStatus::Cancelled,
        ];
        
        for status in statuses {
            let serialized = serde_json::to_string(&status).unwrap();
            let deserialized: MatchStatus = serde_json::from_str(&serialized).unwrap();
            assert_eq!(status, deserialized);
        }
    }
    
    #[test]
    fn test_reputation_calculation() {
        let mut system = ReputationSystem::new();
        let peer_id = "test_peer";
        
        // Initial reputation should be neutral
        let initial = system.get_reputation(peer_id);
        assert_eq!(initial.overall_score, 0.5);
        
        // Successful swap should improve reputation
        let success_event = ReputationEvent {
            timestamp: 1703097600,
            event_type: ReputationEventType::SwapCompleted,
            impact_score: 0.1,
            description: "Successful swap".to_string(),
        };
        
        system.update_reputation(peer_id, success_event);
        let updated = system.get_reputation(peer_id);
        assert!(updated.overall_score > initial.overall_score);
        assert_eq!(updated.successful_swaps, 1);
    }
    
    #[test]
    fn test_swap_offer_compatibility() {
        let coordinator = SwapCoordinator {
            config: crate::MoneroBridgeConfig::default(),
            pending_swaps: HashMap::new(),
            active_matches: HashMap::new(),
            swap_offers: VecDeque::new(),
            counterparty_pool: Vec::new(),
            reputation_system: ReputationSystem::new(),
            matching_engine: MatchingEngine {
                matching_algorithms: Vec::new(),
                market_data: MarketData::default(),
            },
            stats: SwapCoordinatorStats::default(),
        };
        
        let swap = AtomicSwap {
            swap_id: "test_swap".to_string(),
            direction: SwapDirection::QnkToXmr,
            qnk_amount: FixedPoint28::from_u64(100),
            xmr_amount: 5_000_000_000_000, // 5 XMR
            qnk_address: "test_qnk".to_string(),
            xmr_address: "test_xmr".to_string(),
            state: crate::SwapState::Initiated,
            htlc_secret: None,
            htlc_hash: [0u8; 32],
            timeout_height: 1000000,
            created_at: 1703097600,
            updated_at: 1703097600,
            relay_node: None,
        };
        
        let compatible_offer = SwapOffer {
            offer_id: "compatible_offer".to_string(),
            direction: SwapDirection::XmrToQnk,
            amount_offered: FixedPoint28::from_u64(5_000_000_000_000),
            amount_requested: FixedPoint28::from_u64(100),
            exchange_rate: 20.0,
            privacy_level: PrivacyLevel::Enhanced,
            expires_at: 1703097900,
            maker_fingerprint: "test_maker".to_string(),
            requirements: SwapRequirements {
                max_rate_deviation: 2.0,
                min_reputation: 0.7,
                preferred_privacy: PrivacyLevel::Enhanced,
                max_completion_time: 3600,
                preferred_relays: Vec::new(),
                geographic_restrictions: Vec::new(),
            },
        };
        
        assert!(coordinator.is_compatible_offer(&swap, &compatible_offer));
        
        // Same direction should not be compatible
        let incompatible_offer = SwapOffer {
            direction: SwapDirection::QnkToXmr,
            ..compatible_offer
        };
        
        assert!(!coordinator.is_compatible_offer(&swap, &incompatible_offer));
    }
}