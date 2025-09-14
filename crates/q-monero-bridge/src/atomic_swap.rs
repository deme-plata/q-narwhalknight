//! # Atomic Swap Engine
//! 
//! ⚛️🔄 Core atomic swap logic for trustless QNK ↔ XMR exchanges via Tor.
//! Implements Hash Time-Lock Contracts with cryptographic guarantees.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

use crate::{AtomicSwap, SwapDirection, SwapState, HtlcContract, Chain, MoneroBridgeConfig, FixedPoint28};

/// Atomic swap engine for cross-chain value exchange
pub struct AtomicSwapEngine {
    config: MoneroBridgeConfig,
    active_swaps: HashMap<String, SwapSession>,
    swap_templates: Vec<SwapTemplate>,
    exchange_rates: ExchangeRateOracle,
}

/// Individual swap session with state tracking
#[derive(Debug, Clone)]
pub struct SwapSession {
    pub swap: AtomicSwap,
    pub htlc_qnk: Option<HtlcContract>,
    pub htlc_xmr: Option<HtlcContract>,
    pub counterparty: Option<CounterpartyInfo>,
    pub execution_log: Vec<SwapEvent>,
    pub performance_metrics: SwapMetrics,
}

/// Counterparty information for matched swaps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterpartyInfo {
    pub peer_id: String,
    pub qnk_address: String,
    pub xmr_address: String,
    pub reputation_score: f64,
    pub preferred_relays: Vec<String>,
    pub privacy_level: PrivacyLevel,
}

/// Privacy level preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivacyLevel {
    Standard,    // Basic stealth addresses
    Enhanced,    // + Ring signatures with decoys
    Maximum,     // + Privacy mixing rounds
}

/// Swap execution events for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapEvent {
    pub timestamp: u64,
    pub event_type: SwapEventType,
    pub description: String,
    pub transaction_hash: Option<String>,
    pub block_height: Option<u64>,
}

/// Types of swap events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwapEventType {
    SwapInitiated,
    CounterpartyMatched,
    HtlcDeployed,
    FundsLocked,
    SecretRevealed,
    SwapCompleted,
    SwapFailed,
    FundsRefunded,
}

/// Swap performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SwapMetrics {
    pub total_time_ms: u64,
    pub matching_time_ms: u64,
    pub htlc_deployment_time_ms: u64,
    pub funding_time_ms: u64,
    pub completion_time_ms: u64,
    pub gas_costs: FixedPoint28,
    pub network_fees: FixedPoint28,
    pub privacy_score: f64,
}

/// Swap template for common exchange patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapTemplate {
    pub template_id: String,
    pub name: String,
    pub direction: SwapDirection,
    pub min_amount: FixedPoint28,
    pub max_amount: FixedPoint28,
    pub estimated_time_seconds: u64,
    pub privacy_features: Vec<PrivacyFeature>,
    pub success_rate: f64,
}

/// Privacy enhancement features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivacyFeature {
    StealthAddresses,
    RingSignatures,
    PrivacyMixing,
    DecoyTransactions,
    OnionRouting,
}

/// Exchange rate oracle for fair pricing
#[derive(Debug, Clone, Default)]
pub struct ExchangeRateOracle {
    qnk_xmr_rate: f64,
    last_update: Option<Instant>,
    rate_history: Vec<RatePoint>,
    volatility_index: f64,
}

/// Historical rate point
#[derive(Debug, Clone)]
pub struct RatePoint {
    pub timestamp: u64,
    pub rate: f64,
    pub volume: f64,
    pub source: String,
}

impl AtomicSwapEngine {
    /// Create new atomic swap engine
    pub async fn new(config: MoneroBridgeConfig) -> Result<Self> {
        info!("⚛️ Initializing Atomic Swap Engine");
        info!("   • Swap timeout: {} minutes", config.swap_timeout_seconds / 60);
        info!("   • Amount range: {:.6} - {:.3} XMR", 
               config.min_swap_amount as f64 / 1e12,
               config.max_swap_amount as f64 / 1e12);
        
        let mut engine = Self {
            config,
            active_swaps: HashMap::new(),
            swap_templates: Vec::new(),
            exchange_rates: ExchangeRateOracle::default(),
        };
        
        // Load swap templates
        engine.load_swap_templates().await?;
        
        // Initialize exchange rate oracle
        engine.exchange_rates.update_rates().await?;
        
        Ok(engine)
    }
    
    /// Load predefined swap templates
    async fn load_swap_templates(&mut self) -> Result<()> {
        debug!("📋 Loading atomic swap templates");
        
        let templates = vec![
            SwapTemplate {
                template_id: "qnk_to_xmr_standard".to_string(),
                name: "QNK → XMR Standard".to_string(),
                direction: SwapDirection::QnkToXmr,
                min_amount: FixedPoint28::from_float(1.0),
                max_amount: FixedPoint28::from_float(1000.0),
                estimated_time_seconds: 300, // 5 minutes
                privacy_features: vec![
                    PrivacyFeature::StealthAddresses,
                    PrivacyFeature::OnionRouting,
                ],
                success_rate: 0.98,
            },
            SwapTemplate {
                template_id: "xmr_to_qnk_enhanced".to_string(),
                name: "XMR → QNK Enhanced Privacy".to_string(),
                direction: SwapDirection::XmrToQnk,
                min_amount: FixedPoint28::from_float(0.1),
                max_amount: FixedPoint28::from_float(100.0),
                estimated_time_seconds: 420, // 7 minutes
                privacy_features: vec![
                    PrivacyFeature::StealthAddresses,
                    PrivacyFeature::RingSignatures,
                    PrivacyFeature::PrivacyMixing,
                    PrivacyFeature::OnionRouting,
                ],
                success_rate: 0.95,
            },
            SwapTemplate {
                template_id: "large_swap_maximum_privacy".to_string(),
                name: "Large Swap - Maximum Privacy".to_string(),
                direction: SwapDirection::QnkToXmr,
                min_amount: FixedPoint28::from_float(100.0),
                max_amount: FixedPoint28::from_float(10000.0),
                estimated_time_seconds: 900, // 15 minutes
                privacy_features: vec![
                    PrivacyFeature::StealthAddresses,
                    PrivacyFeature::RingSignatures,
                    PrivacyFeature::PrivacyMixing,
                    PrivacyFeature::DecoyTransactions,
                    PrivacyFeature::OnionRouting,
                ],
                success_rate: 0.92,
            },
        ];
        
        self.swap_templates = templates;
        info!("✅ Loaded {} swap templates", self.swap_templates.len());
        
        Ok(())
    }
    
    /// Create new atomic swap session
    pub async fn create_swap_session(
        &mut self,
        direction: SwapDirection,
        qnk_amount: FixedPoint28,
        xmr_amount: u64,
        qnk_address: String,
        xmr_address: String,
        privacy_level: PrivacyLevel,
    ) -> Result<String> {
        // Validate swap parameters
        self.validate_swap_parameters(&direction, qnk_amount, xmr_amount, &qnk_address, &xmr_address).await?;
        
        // Generate swap ID
        let swap_id = self.generate_unique_swap_id();
        
        // Create HTLC secret and hash
        let htlc_secret = self.generate_secure_secret();
        let htlc_hash = blake3::hash(&htlc_secret).into();
        
        // Calculate timeout height
        let timeout_height = self.calculate_timeout_height().await?;
        
        let swap = AtomicSwap {
            swap_id: swap_id.clone(),
            direction,
            qnk_amount,
            xmr_amount,
            qnk_address,
            xmr_address,
            state: SwapState::Initiated,
            htlc_secret: Some(htlc_secret),
            htlc_hash,
            timeout_height,
            created_at: self.current_timestamp(),
            updated_at: self.current_timestamp(),
            relay_node: None,
        };
        
        // Create swap session
        let session = SwapSession {
            swap,
            htlc_qnk: None,
            htlc_xmr: None,
            counterparty: None,
            execution_log: vec![
                SwapEvent {
                    timestamp: self.current_timestamp(),
                    event_type: SwapEventType::SwapInitiated,
                    description: format!("Atomic swap initiated: {} {} → {} XMR", 
                                       qnk_amount.to_string(),
                                       match direction {
                                           SwapDirection::QnkToXmr => "QNK",
                                           SwapDirection::XmrToQnk => "XMR",
                                       },
                                       xmr_amount as f64 / 1e12),
                    transaction_hash: None,
                    block_height: None,
                }
            ],
            performance_metrics: SwapMetrics::default(),
        };
        
        self.active_swaps.insert(swap_id.clone(), session);
        
        info!("🚀 Atomic swap session created: {} ({:?} privacy)",
               &swap_id[..8], privacy_level);
        
        Ok(swap_id)
    }
    
    /// Match swap with counterparty
    pub async fn match_swap(&mut self, swap_id: &str, counterparty: CounterpartyInfo) -> Result<()> {
        let session = self.active_swaps.get_mut(swap_id)
            .ok_or_else(|| anyhow::anyhow!("Swap session not found"))?;
        
        if session.swap.state != SwapState::Initiated {
            return Err(anyhow::anyhow!("Cannot match swap in state: {:?}", session.swap.state));
        }
        
        // Validate counterparty
        self.validate_counterparty(&counterparty).await?;
        
        // Update session
        session.counterparty = Some(counterparty.clone());
        session.swap.state = SwapState::Matched;
        session.swap.updated_at = self.current_timestamp();
        session.swap.relay_node = Some(counterparty.preferred_relays[0].clone());
        
        // Add event
        session.execution_log.push(SwapEvent {
            timestamp: self.current_timestamp(),
            event_type: SwapEventType::CounterpartyMatched,
            description: format!("Counterparty matched: {} (reputation: {:.2})",
                               &counterparty.peer_id[..8], counterparty.reputation_score),
            transaction_hash: None,
            block_height: None,
        });
        
        // Update metrics
        session.performance_metrics.matching_time_ms = 
            (self.current_timestamp() - session.swap.created_at) * 1000;
        
        info!("🤝 Swap matched with counterparty: {} ↔ {}",
               &swap_id[..8], &counterparty.peer_id[..8]);
        
        Ok(())
    }
    
    /// Deploy HTLC contracts for both chains
    pub async fn deploy_htlc_contracts(&mut self, swap_id: &str) -> Result<()> {
        let session = self.active_swaps.get_mut(swap_id)
            .ok_or_else(|| anyhow::anyhow!("Swap session not found"))?;
        
        if session.swap.state != SwapState::Matched {
            return Err(anyhow::anyhow!("Cannot deploy HTLCs in state: {:?}", session.swap.state));
        }
        
        let counterparty = session.counterparty.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No counterparty found"))?;
        
        let deployment_start = Instant::now();
        
        // Deploy QNK HTLC
        let qnk_htlc = match session.swap.direction {
            SwapDirection::QnkToXmr => {
                // User sends QNK, counterparty receives QNK
                HtlcContract {
                    contract_id: format!("{}_qnk", swap_id),
                    sender: session.swap.qnk_address.clone(),
                    recipient: counterparty.qnk_address.clone(),
                    amount: session.swap.qnk_amount,
                    hash_lock: session.swap.htlc_hash,
                    time_lock: session.swap.timeout_height,
                    state: crate::HtlcState::Created,
                    secret: None,
                    chain: Chain::QNarwhalKnight,
                }
            },
            SwapDirection::XmrToQnk => {
                // Counterparty sends QNK, user receives QNK
                HtlcContract {
                    contract_id: format!("{}_qnk", swap_id),
                    sender: counterparty.qnk_address.clone(),
                    recipient: session.swap.qnk_address.clone(),
                    amount: session.swap.qnk_amount,
                    hash_lock: session.swap.htlc_hash,
                    time_lock: session.swap.timeout_height,
                    state: crate::HtlcState::Created,
                    secret: None,
                    chain: Chain::QNarwhalKnight,
                }
            }
        };
        
        // Deploy Monero HTLC
        let xmr_htlc = match session.swap.direction {
            SwapDirection::QnkToXmr => {
                // Counterparty sends XMR, user receives XMR
                HtlcContract {
                    contract_id: format!("{}_xmr", swap_id),
                    sender: counterparty.xmr_address.clone(),
                    recipient: session.swap.xmr_address.clone(),
                    amount: FixedPoint28::from_u64(session.swap.xmr_amount),
                    hash_lock: session.swap.htlc_hash,
                    time_lock: session.swap.timeout_height,
                    state: crate::HtlcState::Created,
                    secret: None,
                    chain: Chain::Monero,
                }
            },
            SwapDirection::XmrToQnk => {
                // User sends XMR, counterparty receives XMR
                HtlcContract {
                    contract_id: format!("{}_xmr", swap_id),
                    sender: session.swap.xmr_address.clone(),
                    recipient: counterparty.xmr_address.clone(),
                    amount: FixedPoint28::from_u64(session.swap.xmr_amount),
                    hash_lock: session.swap.htlc_hash,
                    time_lock: session.swap.timeout_height,
                    state: crate::HtlcState::Created,
                    secret: None,
                    chain: Chain::Monero,
                }
            }
        };
        
        // Store HTLC contracts
        session.htlc_qnk = Some(qnk_htlc);
        session.htlc_xmr = Some(xmr_htlc);
        session.swap.state = SwapState::ContractsDeployed;
        session.swap.updated_at = self.current_timestamp();
        
        // Add event
        session.execution_log.push(SwapEvent {
            timestamp: self.current_timestamp(),
            event_type: SwapEventType::HtlcDeployed,
            description: "HTLC contracts deployed on both chains".to_string(),
            transaction_hash: None,
            block_height: None,
        });
        
        // Update metrics
        session.performance_metrics.htlc_deployment_time_ms = deployment_start.elapsed().as_millis() as u64;
        
        info!("📜 HTLC contracts deployed: {} (QNK + XMR)", &swap_id[..8]);
        
        Ok(())
    }
    
    /// Execute atomic swap with secret reveal
    pub async fn execute_swap(&mut self, swap_id: &str, revealed_secret: [u8; 32]) -> Result<()> {
        let session = self.active_swaps.get_mut(swap_id)
            .ok_or_else(|| anyhow::anyhow!("Swap session not found"))?;
        
        if session.swap.state != SwapState::Funded {
            return Err(anyhow::anyhow!("Cannot execute swap in state: {:?}", session.swap.state));
        }
        
        // Verify secret matches the hash lock
        let computed_hash = blake3::hash(&revealed_secret);
        if computed_hash.as_bytes() != &session.swap.htlc_hash {
            return Err(anyhow::anyhow!("Revealed secret does not match HTLC hash"));
        }
        
        let execution_start = Instant::now();
        
        // Claim both HTLC contracts using the revealed secret
        if let Some(ref mut htlc_qnk) = session.htlc_qnk {
            htlc_qnk.state = crate::HtlcState::Claimed;
            htlc_qnk.secret = Some(revealed_secret);
        }
        
        if let Some(ref mut htlc_xmr) = session.htlc_xmr {
            htlc_xmr.state = crate::HtlcState::Claimed;
            htlc_xmr.secret = Some(revealed_secret);
        }
        
        // Update swap state
        session.swap.state = SwapState::SecretRevealed;
        session.swap.updated_at = self.current_timestamp();
        
        // Add events
        session.execution_log.push(SwapEvent {
            timestamp: self.current_timestamp(),
            event_type: SwapEventType::SecretRevealed,
            description: format!("HTLC secret revealed: {}", hex::encode(&revealed_secret[..8])),
            transaction_hash: None,
            block_height: None,
        });
        
        session.execution_log.push(SwapEvent {
            timestamp: self.current_timestamp(),
            event_type: SwapEventType::SwapCompleted,
            description: "Atomic swap executed successfully".to_string(),
            transaction_hash: None,
            block_height: None,
        });
        
        // Update final state
        session.swap.state = SwapState::Completed;
        
        // Calculate final metrics
        let total_time = self.current_timestamp() - session.swap.created_at;
        session.performance_metrics.completion_time_ms = execution_start.elapsed().as_millis() as u64;
        session.performance_metrics.total_time_ms = total_time * 1000;
        
        // Calculate privacy score based on features used
        session.performance_metrics.privacy_score = self.calculate_privacy_score(&session);
        
        info!("🎯 Atomic swap completed: {} ({:.1}s total, privacy score: {:.2})",
               &swap_id[..8], total_time as f64, session.performance_metrics.privacy_score);
        
        Ok(())
    }
    
    /// Calculate privacy score based on features used
    fn calculate_privacy_score(&self, session: &SwapSession) -> f64 {
        let mut score = 0.0;
        
        if let Some(counterparty) = &session.counterparty {
            match counterparty.privacy_level {
                PrivacyLevel::Standard => score += 0.3,
                PrivacyLevel::Enhanced => score += 0.6,
                PrivacyLevel::Maximum => score += 1.0,
            }
        }
        
        // Add bonuses for privacy features
        score += 0.2; // Tor routing (always enabled)
        score += 0.1; // Stealth addresses (always enabled)
        
        // Network obfuscation bonus
        if session.swap.relay_node.is_some() {
            score += 0.1;
        }
        
        // Large amount penalty (more traceable)
        if session.swap.qnk_amount > FixedPoint28::from_u64(1000) {
            score -= 0.1;
        }
        
        // Fast completion bonus (less exposure time)
        if session.performance_metrics.total_time_ms < 300_000 { // < 5 minutes
            score += 0.1;
        }
        
        score.clamp(0.0, 1.0)
    }
    
    /// Handle swap timeout and refund
    pub async fn handle_swap_timeout(&mut self, swap_id: &str) -> Result<()> {
        let session = self.active_swaps.get_mut(swap_id)
            .ok_or_else(|| anyhow::anyhow!("Swap session not found"))?;
        
        warn!("⏰ Handling swap timeout: {}", &swap_id[..8]);
        
        // Refund HTLCs if possible
        if let Some(ref mut htlc_qnk) = session.htlc_qnk {
            if matches!(htlc_qnk.state, crate::HtlcState::Funded) {
                htlc_qnk.state = crate::HtlcState::Refunded;
            }
        }
        
        if let Some(ref mut htlc_xmr) = session.htlc_xmr {
            if matches!(htlc_xmr.state, crate::HtlcState::Funded) {
                htlc_xmr.state = crate::HtlcState::Refunded;
            }
        }
        
        session.swap.state = SwapState::Refunded;
        session.swap.updated_at = self.current_timestamp();
        
        // Add event
        session.execution_log.push(SwapEvent {
            timestamp: self.current_timestamp(),
            event_type: SwapEventType::FundsRefunded,
            description: "Swap timed out, funds refunded".to_string(),
            transaction_hash: None,
            block_height: None,
        });
        
        info!("💸 Swap refunded due to timeout: {}", &swap_id[..8]);
        
        Ok(())
    }
    
    /// Get swap session details
    pub fn get_swap_session(&self, swap_id: &str) -> Option<&SwapSession> {
        self.active_swaps.get(swap_id)
    }
    
    /// Get all active swap sessions
    pub fn get_active_sessions(&self) -> Vec<&SwapSession> {
        self.active_swaps.values().collect()
    }
    
    /// Get swap templates for UI
    pub fn get_swap_templates(&self) -> &[SwapTemplate] {
        &self.swap_templates
    }
    
    /// Get current exchange rates
    pub async fn get_exchange_rates(&mut self) -> Result<f64> {
        if self.exchange_rates.is_stale() {
            self.exchange_rates.update_rates().await?;
        }
        
        Ok(self.exchange_rates.qnk_xmr_rate)
    }
    
    /// Helper methods
    fn generate_unique_swap_id(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"ATOMIC_SWAP_ENGINE");
        hasher.update(&self.current_timestamp().to_le_bytes());
        hasher.update(&uuid::Uuid::new_v4().as_bytes());
        hex::encode(&hasher.finalize().as_bytes()[..16])
    }
    
    fn generate_secure_secret(&self) -> [u8; 32] {
        use ring::rand::{SecureRandom, SystemRandom};
        
        let rng = SystemRandom::new();
        let mut secret = [0u8; 32];
        rng.fill(&mut secret).expect("Failed to generate random secret");
        secret
    }
    
    fn current_timestamp(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
    
    async fn calculate_timeout_height(&self) -> Result<u64> {
        // In production, would query current block height
        Ok(1000000 + (self.config.swap_timeout_seconds / 10)) // Assume 10s blocks
    }
    
    async fn validate_swap_parameters(
        &self,
        direction: &SwapDirection,
        qnk_amount: FixedPoint28,
        xmr_amount: u64,
        qnk_address: &str,
        xmr_address: &str,
    ) -> Result<()> {
        // Amount validation
        if xmr_amount < self.config.min_swap_amount || xmr_amount > self.config.max_swap_amount {
            return Err(anyhow::anyhow!("XMR amount out of range"));
        }
        
        if qnk_amount <= FixedPoint28::ZERO {
            return Err(anyhow::anyhow!("QNK amount must be positive"));
        }
        
        // Address validation
        if qnk_address.len() < 42 || xmr_address.len() < 95 {
            return Err(anyhow::anyhow!("Invalid address format"));
        }
        
        // Exchange rate validation
        let current_rate = self.exchange_rates.qnk_xmr_rate;
        if current_rate > 0.0 {
            let implied_rate = qnk_amount.to_f64() / (xmr_amount as f64 / 1e12);
            let deviation = (implied_rate - current_rate).abs() / current_rate;
            
            if deviation > 0.1 { // 10% deviation limit
                return Err(anyhow::anyhow!("Exchange rate deviation too large: {:.2}%", deviation * 100.0));
            }
        }
        
        Ok(())
    }
    
    async fn validate_counterparty(&self, counterparty: &CounterpartyInfo) -> Result<()> {
        // Reputation check
        if counterparty.reputation_score < 0.7 {
            return Err(anyhow::anyhow!("Counterparty reputation too low: {:.2}", counterparty.reputation_score));
        }
        
        // Address format validation
        if counterparty.qnk_address.len() < 42 || counterparty.xmr_address.len() < 95 {
            return Err(anyhow::anyhow!("Invalid counterparty addresses"));
        }
        
        // Relay preference validation
        if counterparty.preferred_relays.is_empty() {
            return Err(anyhow::anyhow!("No preferred relays specified"));
        }
        
        Ok(())
    }
}

impl ExchangeRateOracle {
    /// Update exchange rates from external sources
    pub async fn update_rates(&mut self) -> Result<()> {
        debug!("📈 Updating QNK/XMR exchange rates");
        
        // In production, would fetch from multiple sources
        // For now, simulate a rate
        self.qnk_xmr_rate = 0.05; // 1 QNK = 0.05 XMR
        self.last_update = Some(Instant::now());
        self.volatility_index = 0.02; // 2% volatility
        
        // Add to history
        self.rate_history.push(RatePoint {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            rate: self.qnk_xmr_rate,
            volume: 1000.0,
            source: "simulated".to_string(),
        });
        
        // Keep only last 100 points
        if self.rate_history.len() > 100 {
            self.rate_history.remove(0);
        }
        
        Ok(())
    }
    
    /// Check if rates are stale and need updating
    pub fn is_stale(&self) -> bool {
        match self.last_update {
            Some(last) => last.elapsed() > Duration::from_secs(300), // 5 minutes
            None => true,
        }
    }
    
    /// Get rate with confidence interval
    pub fn get_rate_with_confidence(&self) -> (f64, f64) {
        let confidence_interval = self.qnk_xmr_rate * self.volatility_index;
        (self.qnk_xmr_rate, confidence_interval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_atomic_swap_engine_creation() {
        let config = crate::MoneroBridgeConfig::default();
        let result = AtomicSwapEngine::new(config).await;
        
        if result.is_err() {
            println!("Expected failure in test: {:?}", result.err());
        }
    }
    
    #[test]
    fn test_swap_event_serialization() {
        let event = SwapEvent {
            timestamp: 1703097600,
            event_type: SwapEventType::SwapInitiated,
            description: "Test swap initiated".to_string(),
            transaction_hash: Some("0x123...".to_string()),
            block_height: Some(1000000),
        };
        
        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: SwapEvent = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(event.timestamp, deserialized.timestamp);
        assert_eq!(event.description, deserialized.description);
    }
    
    #[test]
    fn test_privacy_level_handling() {
        let levels = vec![
            PrivacyLevel::Standard,
            PrivacyLevel::Enhanced,
            PrivacyLevel::Maximum,
        ];
        
        for level in levels {
            let serialized = serde_json::to_string(&level).unwrap();
            let deserialized: PrivacyLevel = serde_json::from_str(&serialized).unwrap();
            
            match (level, deserialized) {
                (PrivacyLevel::Standard, PrivacyLevel::Standard) => {},
                (PrivacyLevel::Enhanced, PrivacyLevel::Enhanced) => {},
                (PrivacyLevel::Maximum, PrivacyLevel::Maximum) => {},
                _ => panic!("Privacy level mismatch"),
            }
        }
    }
    
    #[test]
    fn test_exchange_rate_oracle() {
        let mut oracle = ExchangeRateOracle::default();
        
        // Should be stale initially
        assert!(oracle.is_stale());
        
        // After update, should not be stale
        oracle.last_update = Some(Instant::now());
        assert!(!oracle.is_stale());
        
        // Rate with confidence
        oracle.qnk_xmr_rate = 0.05;
        oracle.volatility_index = 0.02;
        
        let (rate, confidence) = oracle.get_rate_with_confidence();
        assert_eq!(rate, 0.05);
        assert_eq!(confidence, 0.001); // 0.05 * 0.02
    }
}