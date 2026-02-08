//! Oracle Integration Module for QNO Prediction Resolution
//!
//! Provides oracle data feeds for resolving prediction stakes.
//! Supports multiple oracle sources:
//! - Simulated oracle (for testing/development)
//! - Chainlink (production-ready stub)
//! - Pyth Network (production-ready stub)
//!
//! v1.4.3-beta: Initial implementation

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

#[cfg(not(target_os = "windows"))]
use q_storage::qno_storage::{OutcomeType, PredictionOutcome};

// Windows stubs for qno_storage types (RocksDB not available)
#[cfg(target_os = "windows")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum OutcomeType {
    GasFee,
    BlockTime,
    NetworkLoad,
    ValidatorUptime,
    CrossChain,
    DefiTvl,
    Custom(String),
}

#[cfg(target_os = "windows")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PredictionOutcome {
    pub id: String,
    pub domain: String,
    pub outcome_type: OutcomeType,
    pub predicted_value: f64,
    pub actual_value: f64,
    pub timestamp: u64,
    pub confidence_threshold: f64,
    pub oracle_signature: Vec<u8>,
}

// ============================================================================
// Oracle Provider Trait
// ============================================================================

/// Enum-based oracle provider (avoids async trait object issues)
#[derive(Clone)]
pub enum OracleProviderType {
    Simulated(SimulatedOracle),
    Chainlink(ChainlinkOracle),
    Pyth(PythOracle),
}

impl OracleProviderType {
    /// Get the name of this oracle provider
    pub fn name(&self) -> &str {
        match self {
            OracleProviderType::Simulated(_) => "simulated",
            OracleProviderType::Chainlink(_) => "chainlink",
            OracleProviderType::Pyth(_) => "pyth",
        }
    }

    /// Fetch the latest value for a prediction domain
    pub async fn fetch_value(&self, domain: &str, outcome_type: &OutcomeType) -> Result<f64> {
        match self {
            OracleProviderType::Simulated(p) => p.fetch_value(domain, outcome_type).await,
            OracleProviderType::Chainlink(p) => p.fetch_value(domain, outcome_type).await,
            OracleProviderType::Pyth(p) => p.fetch_value(domain, outcome_type).await,
        }
    }

    /// Get the confidence level of this oracle (0.0 to 1.0)
    pub fn confidence(&self) -> f64 {
        match self {
            OracleProviderType::Simulated(p) => p.confidence(),
            OracleProviderType::Chainlink(p) => p.confidence(),
            OracleProviderType::Pyth(p) => p.confidence(),
        }
    }

    /// Check if this oracle supports a given domain
    pub fn supports_domain(&self, domain: &str) -> bool {
        match self {
            OracleProviderType::Simulated(p) => p.supports_domain(domain),
            OracleProviderType::Chainlink(p) => p.supports_domain(domain),
            OracleProviderType::Pyth(p) => p.supports_domain(domain),
        }
    }
}

// ============================================================================
// Oracle Data Types
// ============================================================================

/// Oracle feed configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleFeedConfig {
    pub domain: String,
    pub outcome_type: OutcomeType,
    pub update_interval_secs: u64,
    pub min_confidence: f64,
    pub providers: Vec<String>,  // List of provider names to use
}

/// Aggregated oracle value from multiple sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedOracleValue {
    pub domain: String,
    pub value: f64,
    pub confidence: f64,
    pub timestamp: u64,
    pub sources: Vec<OracleSource>,
}

/// Individual oracle source contribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleSource {
    pub provider: String,
    pub value: f64,
    pub confidence: f64,
    pub timestamp: u64,
}

// ============================================================================
// Simulated Oracle (Testing/Development)
// ============================================================================

/// Simulated oracle for testing and development
pub struct SimulatedOracle {
    /// Base values for each domain (used to generate realistic variations)
    base_values: HashMap<String, f64>,
    /// Volatility factor for each domain
    volatility: HashMap<String, f64>,
}

impl SimulatedOracle {
    pub fn new() -> Self {
        let mut base_values = HashMap::new();
        let mut volatility = HashMap::new();

        // Gas fees: typically 20-100 gwei
        base_values.insert("gas-fees".to_string(), 45.0);
        volatility.insert("gas-fees".to_string(), 0.3);  // 30% volatility

        // Block time: typically 2-15 seconds
        base_values.insert("block-time".to_string(), 12.0);
        volatility.insert("block-time".to_string(), 0.1);  // 10% volatility

        // Network load: 0-100%
        base_values.insert("network-load".to_string(), 65.0);
        volatility.insert("network-load".to_string(), 0.25);  // 25% volatility

        // Validator uptime: 95-100%
        base_values.insert("validator-uptime".to_string(), 98.5);
        volatility.insert("validator-uptime".to_string(), 0.02);  // 2% volatility

        // Cross-chain success rate: 90-100%
        base_values.insert("cross-chain".to_string(), 96.0);
        volatility.insert("cross-chain".to_string(), 0.05);  // 5% volatility

        // DeFi TVL: billions
        base_values.insert("defi-tvl".to_string(), 45_000_000_000.0);
        volatility.insert("defi-tvl".to_string(), 0.15);  // 15% volatility

        Self { base_values, volatility }
    }

    /// Generate a simulated value with realistic variation
    fn generate_value(&self, domain: &str) -> f64 {
        let base = self.base_values.get(domain).copied().unwrap_or(100.0);
        let vol = self.volatility.get(domain).copied().unwrap_or(0.1);

        // Generate random variation using simple deterministic noise
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Use timestamp to create pseudo-random variation
        let noise_factor = ((now % 1000) as f64 / 1000.0) * 2.0 - 1.0;  // -1 to 1
        let variation = base * vol * noise_factor;

        (base + variation).max(0.0)  // Ensure non-negative
    }
}

impl Default for SimulatedOracle {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SimulatedOracle {
    fn clone(&self) -> Self {
        Self {
            base_values: self.base_values.clone(),
            volatility: self.volatility.clone(),
        }
    }
}

impl SimulatedOracle {
    pub async fn fetch_value(&self, domain: &str, _outcome_type: &OutcomeType) -> Result<f64> {
        if !self.supports_domain(domain) {
            return Err(anyhow!("Domain {} not supported by simulated oracle", domain));
        }

        let value = self.generate_value(domain);
        debug!("🔮 [Simulated Oracle] {} = {:.4}", domain, value);
        Ok(value)
    }

    pub fn confidence(&self) -> f64 {
        0.75  // Simulated oracle has moderate confidence
    }

    pub fn supports_domain(&self, domain: &str) -> bool {
        self.base_values.contains_key(domain)
    }
}

// ============================================================================
// Chainlink Oracle (Production Stub)
// ============================================================================

/// Chainlink oracle integration (stub for production)
pub struct ChainlinkOracle {
    /// RPC endpoint for Chainlink price feeds
    rpc_endpoint: String,
    /// Feed addresses for each domain
    feed_addresses: HashMap<String, String>,
}

impl ChainlinkOracle {
    pub fn new(rpc_endpoint: String) -> Self {
        let mut feed_addresses = HashMap::new();

        // These would be actual Chainlink feed addresses on mainnet
        // For now, using placeholder addresses
        feed_addresses.insert("gas-fees".to_string(), "0x169E633A2D1E6c10dD91238Ba11c4A708dfEF37C".to_string());

        Self { rpc_endpoint, feed_addresses }
    }

    pub async fn fetch_value(&self, domain: &str, _outcome_type: &OutcomeType) -> Result<f64> {
        // TODO: Implement actual Chainlink integration
        // This would involve:
        // 1. Connect to Ethereum RPC
        // 2. Call the Chainlink Aggregator contract
        // 3. Parse the latestRoundData response

        warn!("⚠️ [Chainlink] Integration not yet implemented, returning stub value");
        Err(anyhow!("Chainlink integration not yet implemented"))
    }

    pub fn confidence(&self) -> f64 {
        0.95  // Chainlink is highly reliable
    }

    pub fn supports_domain(&self, domain: &str) -> bool {
        self.feed_addresses.contains_key(domain)
    }
}

impl Clone for ChainlinkOracle {
    fn clone(&self) -> Self {
        Self {
            rpc_endpoint: self.rpc_endpoint.clone(),
            feed_addresses: self.feed_addresses.clone(),
        }
    }
}

// ============================================================================
// Pyth Network Oracle (Production Stub)
// ============================================================================

/// Pyth Network oracle integration (stub for production)
pub struct PythOracle {
    /// Pyth price service endpoint
    price_service_endpoint: String,
    /// Price feed IDs for each domain
    feed_ids: HashMap<String, String>,
}

impl PythOracle {
    pub fn new(endpoint: String) -> Self {
        let mut feed_ids = HashMap::new();

        // These would be actual Pyth feed IDs
        // For now, using placeholder IDs
        feed_ids.insert("gas-fees".to_string(), "0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace".to_string());

        Self {
            price_service_endpoint: endpoint,
            feed_ids,
        }
    }

    pub async fn fetch_value(&self, domain: &str, _outcome_type: &OutcomeType) -> Result<f64> {
        // TODO: Implement actual Pyth integration
        // This would involve:
        // 1. Connect to Pyth price service
        // 2. Fetch the latest price update
        // 3. Verify the price update signature
        // 4. Parse the price data

        warn!("⚠️ [Pyth] Integration not yet implemented, returning stub value");
        Err(anyhow!("Pyth integration not yet implemented"))
    }

    pub fn confidence(&self) -> f64 {
        0.93  // Pyth is highly reliable
    }

    pub fn supports_domain(&self, domain: &str) -> bool {
        self.feed_ids.contains_key(domain)
    }
}

impl Clone for PythOracle {
    fn clone(&self) -> Self {
        Self {
            price_service_endpoint: self.price_service_endpoint.clone(),
            feed_ids: self.feed_ids.clone(),
        }
    }
}

// ============================================================================
// Oracle Aggregator
// ============================================================================

/// Aggregates data from multiple oracle providers
pub struct OracleAggregator {
    providers: Vec<OracleProviderType>,
    /// Minimum number of oracle sources required for aggregation
    min_sources: usize,
    /// Whether to use simulated oracle as fallback
    use_simulated_fallback: bool,
    /// v2.4.9-beta: Ed25519 signing key for oracle attestations
    signing_key: Option<Arc<ed25519_dalek::SigningKey>>,
}

impl OracleAggregator {
    pub fn new() -> Self {
        Self {
            providers: vec![],
            min_sources: 1,
            use_simulated_fallback: true,
            signing_key: None,
        }
    }

    /// v2.4.9-beta: Set signing key for oracle attestations
    pub fn with_signing_key(mut self, key: Arc<ed25519_dalek::SigningKey>) -> Self {
        self.signing_key = Some(key);
        self
    }

    /// Create with default providers (simulated for dev, production for mainnet)
    pub fn with_defaults(is_production: bool) -> Self {
        let mut aggregator = Self::new();

        if is_production {
            // Add production oracles
            aggregator.add_provider(OracleProviderType::Chainlink(ChainlinkOracle::new(
                "https://eth-mainnet.g.alchemy.com/v2/demo".to_string()
            )));
            aggregator.add_provider(OracleProviderType::Pyth(PythOracle::new(
                "https://hermes.pyth.network".to_string()
            )));
            aggregator.use_simulated_fallback = true;  // Use simulated as backup
            aggregator.min_sources = 2;  // Require multiple sources in production
        } else {
            // Development mode - use simulated oracle
            aggregator.add_provider(OracleProviderType::Simulated(SimulatedOracle::new()));
            aggregator.min_sources = 1;
        }

        aggregator
    }

    /// Add a provider to the aggregator
    pub fn add_provider(&mut self, provider: OracleProviderType) {
        info!("📡 [Oracle] Added provider: {}", provider.name());
        self.providers.push(provider);
    }

    /// Fetch and aggregate values from all providers
    pub async fn fetch_aggregated(&self, domain: &str, outcome_type: &OutcomeType) -> Result<AggregatedOracleValue> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut sources = Vec::new();
        let mut values_and_weights = Vec::new();

        // Fetch from all providers
        for provider in &self.providers {
            if !provider.supports_domain(domain) {
                continue;
            }

            match provider.fetch_value(domain, outcome_type).await {
                Ok(value) => {
                    let confidence = provider.confidence();
                    sources.push(OracleSource {
                        provider: provider.name().to_string(),
                        value,
                        confidence,
                        timestamp: now,
                    });
                    values_and_weights.push((value, confidence));
                }
                Err(e) => {
                    warn!("⚠️ [Oracle] {} failed for {}: {}", provider.name(), domain, e);
                }
            }
        }

        // Fallback to simulated if needed
        if sources.is_empty() && self.use_simulated_fallback {
            let simulated = SimulatedOracle::new();
            if let Ok(value) = simulated.fetch_value(domain, outcome_type).await {
                sources.push(OracleSource {
                    provider: "simulated-fallback".to_string(),
                    value,
                    confidence: simulated.confidence(),
                    timestamp: now,
                });
                values_and_weights.push((value, simulated.confidence()));
            }
        }

        if sources.len() < self.min_sources {
            return Err(anyhow!(
                "Insufficient oracle sources: got {}, need {}",
                sources.len(),
                self.min_sources
            ));
        }

        // Weighted average aggregation
        let total_weight: f64 = values_and_weights.iter().map(|(_, w)| w).sum();
        let weighted_sum: f64 = values_and_weights.iter().map(|(v, w)| v * w).sum();
        let aggregated_value = weighted_sum / total_weight;

        // Combined confidence (product of individual confidences normalized)
        let combined_confidence = sources.iter()
            .map(|s| s.confidence)
            .fold(1.0, |acc, c| acc * c)
            .powf(1.0 / sources.len() as f64);

        info!("📊 [Oracle] Aggregated {} = {:.4} (confidence: {:.2}%, sources: {})",
              domain, aggregated_value, combined_confidence * 100.0, sources.len());

        Ok(AggregatedOracleValue {
            domain: domain.to_string(),
            value: aggregated_value,
            confidence: combined_confidence,
            timestamp: now,
            sources,
        })
    }

    /// Create a PredictionOutcome from aggregated oracle data
    /// v2.4.9-beta: Now signs oracle outcomes with Ed25519 for cryptographic attestation
    pub async fn create_outcome(&self, domain: &str, outcome_type: OutcomeType) -> Result<PredictionOutcome> {
        use ed25519_dalek::Signer;

        let aggregated = self.fetch_aggregated(domain, &outcome_type).await?;
        let outcome_id = uuid::Uuid::new_v4().to_string();

        // v2.4.9-beta: Create canonical data to sign (id + domain + value + timestamp)
        let oracle_signature = if let Some(ref signing_key) = self.signing_key {
            let mut sign_data = Vec::with_capacity(128);
            sign_data.extend_from_slice(outcome_id.as_bytes());
            sign_data.extend_from_slice(domain.as_bytes());
            sign_data.extend_from_slice(&aggregated.value.to_le_bytes());
            sign_data.extend_from_slice(&aggregated.timestamp.to_le_bytes());
            let signature = signing_key.sign(&sign_data);
            debug!("🔐 Signed oracle outcome {} with Ed25519 ({} bytes)",
                   outcome_id, signature.to_bytes().len());
            signature.to_bytes().to_vec()
        } else {
            warn!("⚠️ No signing key set - oracle outcome {} unsigned", outcome_id);
            vec![]
        };

        Ok(PredictionOutcome {
            id: outcome_id,
            domain: domain.to_string(),
            outcome_type,
            predicted_value: 0.0,  // Not applicable for oracle
            actual_value: aggregated.value,
            timestamp: aggregated.timestamp,
            confidence_threshold: aggregated.confidence,
            oracle_signature,
        })
    }
}

impl Default for OracleAggregator {
    fn default() -> Self {
        Self::with_defaults(false)  // Default to development mode
    }
}

// ============================================================================
// Oracle Service
// ============================================================================

/// Oracle service for managing periodic updates
pub struct OracleService {
    aggregator: Arc<OracleAggregator>,
    /// Feed configurations for each domain
    feeds: RwLock<Vec<OracleFeedConfig>>,
    /// Latest values cache
    latest_values: RwLock<HashMap<String, AggregatedOracleValue>>,
}

impl OracleService {
    pub fn new(is_production: bool) -> Self {
        Self {
            aggregator: Arc::new(OracleAggregator::with_defaults(is_production)),
            feeds: RwLock::new(Self::default_feeds()),
            latest_values: RwLock::new(HashMap::new()),
        }
    }

    fn default_feeds() -> Vec<OracleFeedConfig> {
        vec![
            OracleFeedConfig {
                domain: "gas-fees".to_string(),
                outcome_type: OutcomeType::GasFee,
                update_interval_secs: 300,  // 5 minutes
                min_confidence: 0.7,
                providers: vec!["chainlink".to_string(), "simulated".to_string()],
            },
            OracleFeedConfig {
                domain: "block-time".to_string(),
                outcome_type: OutcomeType::BlockTime,
                update_interval_secs: 600,  // 10 minutes
                min_confidence: 0.7,
                providers: vec!["simulated".to_string()],
            },
            OracleFeedConfig {
                domain: "network-load".to_string(),
                outcome_type: OutcomeType::NetworkLoad,
                update_interval_secs: 300,
                min_confidence: 0.7,
                providers: vec!["simulated".to_string()],
            },
            OracleFeedConfig {
                domain: "validator-uptime".to_string(),
                outcome_type: OutcomeType::ValidatorUptime,
                update_interval_secs: 900,  // 15 minutes
                min_confidence: 0.8,
                providers: vec!["simulated".to_string()],
            },
            OracleFeedConfig {
                domain: "cross-chain".to_string(),
                outcome_type: OutcomeType::CrossChain,
                update_interval_secs: 600,
                min_confidence: 0.7,
                providers: vec!["simulated".to_string()],
            },
            OracleFeedConfig {
                domain: "defi-tvl".to_string(),
                outcome_type: OutcomeType::DefiTvl,
                update_interval_secs: 3600,  // 1 hour
                min_confidence: 0.6,
                providers: vec!["simulated".to_string()],
            },
        ]
    }

    /// Fetch latest values for all feeds
    pub async fn update_all_feeds(&self) -> Vec<(String, Result<AggregatedOracleValue>)> {
        let feeds = self.feeds.read().await.clone();
        let mut results = Vec::new();

        for feed in feeds {
            let result = self.aggregator.fetch_aggregated(&feed.domain, &feed.outcome_type).await;

            if let Ok(ref value) = result {
                let mut cache = self.latest_values.write().await;
                cache.insert(feed.domain.clone(), value.clone());
            }

            results.push((feed.domain, result));
        }

        results
    }

    /// Get the latest cached value for a domain
    pub async fn get_latest(&self, domain: &str) -> Option<AggregatedOracleValue> {
        self.latest_values.read().await.get(domain).cloned()
    }

    /// Create outcomes for all domains (for resolution)
    pub async fn create_all_outcomes(&self) -> Vec<PredictionOutcome> {
        let feeds = self.feeds.read().await.clone();
        let mut outcomes = Vec::new();

        for feed in feeds {
            match self.aggregator.create_outcome(&feed.domain, feed.outcome_type.clone()).await {
                Ok(outcome) => outcomes.push(outcome),
                Err(e) => warn!("⚠️ [Oracle] Failed to create outcome for {}: {}", feed.domain, e),
            }
        }

        outcomes
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simulated_oracle() {
        let oracle = SimulatedOracle::new();

        assert!(oracle.supports_domain("gas-fees"));
        assert!(oracle.supports_domain("block-time"));
        assert!(!oracle.supports_domain("unknown-domain"));

        let value = oracle.fetch_value("gas-fees", &OutcomeType::GasFee).await.unwrap();
        assert!(value > 0.0);
    }

    #[tokio::test]
    async fn test_oracle_aggregator() {
        let aggregator = OracleAggregator::with_defaults(false);

        let result = aggregator.fetch_aggregated("gas-fees", &OutcomeType::GasFee).await;
        assert!(result.is_ok());

        let aggregated = result.unwrap();
        assert_eq!(aggregated.domain, "gas-fees");
        assert!(aggregated.value > 0.0);
        assert!(aggregated.confidence > 0.0);
        assert!(!aggregated.sources.is_empty());
    }

    #[tokio::test]
    async fn test_create_outcome() {
        let aggregator = OracleAggregator::with_defaults(false);

        let outcome = aggregator.create_outcome("gas-fees", OutcomeType::GasFee).await.unwrap();
        assert_eq!(outcome.domain, "gas-fees");
        assert!(outcome.actual_value > 0.0);
    }

    #[tokio::test]
    async fn test_oracle_service() {
        let service = OracleService::new(false);

        let results = service.update_all_feeds().await;
        assert!(!results.is_empty());

        // Check we can get a cached value
        let cached = service.get_latest("gas-fees").await;
        assert!(cached.is_some());
    }
}
