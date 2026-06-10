//! Quantum-Enhanced DEX Types
//!
//! Type definitions for the quantum-secured decentralized exchange
//! with physics-inspired properties and post-quantum security

use bigdecimal::BigDecimal;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Quantum state of a trading entity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantumState {
    /// Superposition - exists in multiple states simultaneously
    Superposition,
    /// Collapsed - observed and fixed state
    Collapsed,
    /// Entangled - correlated with another entity
    Entangled,
}

/// Wave interference patterns in trading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WavePattern {
    Constructive, // Reinforcing patterns
    Destructive,  // Canceling patterns
    Neutral,      // No interference
}

impl Default for WavePattern {
    fn default() -> Self {
        WavePattern::Neutral
    }
}

/// Quantum privacy tiers for trading
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantumPrivacyTier {
    /// Standard trading (minimal privacy)
    Basic = 0,
    /// Enhanced privacy with basic Tor routing
    Enhanced = 1,
    /// Maximum privacy with ZK proofs and multi-hop Tor
    Maximum = 2,
    /// Quantum-level privacy with post-quantum cryptography
    Quantum = 3,
}

/// Quantum price feed with uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPriceFeed {
    pub symbol: String,
    pub price: BigDecimal,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub quantum_uncertainty: BigDecimal,
    pub wave_function_collapsed: bool,
    pub entanglement_strength: f64,
}

/// Quantum DEX physics parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumDexParameters {
    // Physics constants
    pub planck_constant: BigDecimal,
    pub golden_ratio: BigDecimal,
    pub euler_constant: BigDecimal,
    pub pi_constant: BigDecimal,

    // Quantum-specific trading parameters
    pub uncertainty_principle_factor: f64,
    pub wave_collapse_threshold: f64,
    pub entanglement_strength: f64,
    pub decoherence_time_seconds: u64,

    // Risk management parameters
    pub max_leverage: f64,
    /// Liquidation threshold in basis points (e.g., 8000 = 80%)
    pub liquidation_threshold_bps: u16,
    /// Maximum slippage protection in basis points (e.g., 50 = 0.5%)
    pub slippage_protection_bps: u16,
}

impl Default for QuantumDexParameters {
    fn default() -> Self {
        Self {
            planck_constant: "0.00000000000000000000000000000000066260701".parse().unwrap(),
            golden_ratio: "1.618033988749895".parse().unwrap(),
            euler_constant: "2.718281828459045".parse().unwrap(),
            pi_constant: "3.141592653589793".parse().unwrap(),
            uncertainty_principle_factor: 0.1618,
            wave_collapse_threshold: 0.05,
            entanglement_strength: 0.707,
            decoherence_time_seconds: 300,
            max_leverage: 10.0,
            liquidation_threshold_bps: 8000, // 80%
            slippage_protection_bps: 50,     // 0.5%
        }
    }
}

/// Quantum-enhanced token definition with physics properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumToken {
    pub symbol: String,
    pub name: String,
    pub decimals: u8,
    pub contract_address: Option<String>,
    pub total_supply: BigDecimal,
    pub quantum_secured: bool,
    pub privacy_enabled: bool,
    pub zk_proofs_required: bool,
    pub created_at: DateTime<Utc>,

    // Market data fields
    pub price_usd: Option<BigDecimal>,
    pub market_cap: Option<BigDecimal>,
    pub circulating_supply: Option<BigDecimal>,
    pub volume_24h: Option<BigDecimal>,

    // Token metadata
    pub description: Option<String>,
    pub logo_url: Option<String>,
    pub website: Option<String>,
    pub tags: Vec<String>,
    pub address: Option<String>,
    pub quantum_signature_verified: bool,

    // Quantum-specific properties
    pub quantum_volatility: BigDecimal,
    pub wave_function_state: QuantumState,
    pub entanglement_pairs: Vec<String>,
    pub defi_protocols: Vec<String>,
}

/// Alias for backward compatibility
pub type QuantumTokenInfo = QuantumToken;

/// Quantum trading pair with physics-based properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumTradingPair {
    pub id: String,
    pub pair_id: String, // Alias for id
    pub base_token: String,
    pub quote_token: String,
    pub base_address: Option<String>,
    pub quote_address: Option<String>,
    pub exchange: String,
    pub price: BigDecimal,
    pub volume_24h: BigDecimal,
    pub fee_rate: u16,        // in basis points
    pub fee_tier: BigDecimal, // Alternative fee representation
    pub min_trade_size: BigDecimal,
    pub max_trade_size: BigDecimal,
    pub liquidity: BigDecimal,
    pub quantum_secured: bool,
    pub privacy_tier: QuantumPrivacyTier,
    pub zk_proof_required: bool,
    pub created_at: DateTime<Utc>,
    pub active: bool,

    // Quantum pair properties
    pub quantum_correlation: f64,
    pub wave_interference_pattern: WavePattern,
    pub price_uncertainty: BigDecimal,
    pub quantum_liquidity_depth: BigDecimal,
    pub entangled_state: bool,
}

/// Quantum trade request with post-quantum security
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumTradeRequest {
    pub user: String, // Alias for trader_id
    pub trader_id: String,
    pub pair_id: String,
    pub side: TradeSide,
    pub amount: BigDecimal,
    pub price: Option<BigDecimal>, // None for market orders
    pub order_type: OrderType,
    pub privacy_level: QuantumPrivacyTier,
    pub zk_proof_required: bool,
    /// Maximum slippage in basis points (e.g., 100 = 1%, 50 = 0.5%)
    pub max_slippage_bps: u16,
    pub expires_at: Option<DateTime<Utc>>,

    // Quantum-specific fields
    pub quantum_signature: Vec<u8>, // Post-quantum cryptographic signature
    pub entanglement_proof: Option<Vec<u8>>,
}

/// Trade side enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Order type enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    TakeProfit,
}

/// Quantum trade result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumTradeResult {
    pub trade_id: String,
    pub trader_id: String,
    pub pair_id: String,
    pub side: TradeSide,
    pub amount_filled: BigDecimal,
    pub price: BigDecimal,
    pub fees_paid: BigDecimal,
    pub privacy_level: QuantumPrivacyTier,
    pub zk_proof: Option<QuantumZkProof>,
    pub tor_circuit_id: Option<String>,
    pub executed_at: DateTime<Utc>,
    pub block_height: Option<u64>,
    pub quantum_signature: Option<Vec<u8>>,
}

/// Zero-knowledge proof for trades
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumZkProof {
    pub proof_data: Vec<u8>,
    pub public_inputs: Vec<String>,
    pub circuit_type: ZkCircuitType,
    pub generated_at: DateTime<Utc>,
}

/// Types of ZK circuits supported
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ZkCircuitType {
    TradeValidation,
    LiquidityProof,
    BalanceProof,
    PrivacyMixing,
}

/// Quantum liquidity request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumLiquidityRequest {
    pub provider_id: String,
    pub pair_id: String,
    pub token_a_amount: BigDecimal,
    pub token_b_amount: BigDecimal,
    pub min_shares: BigDecimal,
    pub privacy_level: QuantumPrivacyTier,
    pub lock_period: Option<u64>, // seconds
}

/// Quantum liquidity position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumLiquidityPosition {
    pub position_id: String,
    pub provider_id: String,
    pub pair_id: String,
    pub shares: BigDecimal,
    pub token_a_amount: BigDecimal,
    pub token_b_amount: BigDecimal,
    pub fees_earned: BigDecimal,
    pub privacy_level: QuantumPrivacyTier,
    pub zk_proof: Option<QuantumZkProof>,
    pub created_at: DateTime<Utc>,
    pub locked_until: Option<DateTime<Utc>>,
}

/// Quantum market data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMarketData {
    pub pair_id: String,
    pub current_price: BigDecimal,
    pub volume_24h: BigDecimal,
    pub liquidity: BigDecimal,
    /// Price change in 24h in basis points (e.g., 550 = 5.5%, -200 = -2%)
    pub price_change_24h_bps: i32,
    pub high_24h: BigDecimal,
    pub low_24h: BigDecimal,
    pub trades_count: u64,
    pub quantum_signature: Option<Vec<u8>>,
    pub privacy_stats: QuantumPrivacyStats,
    pub timestamp: DateTime<Utc>,
}

impl Default for QuantumMarketData {
    fn default() -> Self {
        Self {
            pair_id: String::new(),
            current_price: BigDecimal::from(0),
            volume_24h: BigDecimal::from(0),
            liquidity: BigDecimal::from(0),
            price_change_24h_bps: 0,
            high_24h: BigDecimal::from(0),
            low_24h: BigDecimal::from(0),
            trades_count: 0,
            quantum_signature: None,
            privacy_stats: QuantumPrivacyStats::default(),
            timestamp: Utc::now(),
        }
    }
}

/// Privacy statistics for quantum trading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPrivacyStats {
    pub total_private_trades: u64,
    /// ZK proof success rate in basis points (e.g., 10000 = 100%, 9950 = 99.5%)
    pub zk_proof_success_rate_bps: u16,
    pub tor_circuits_used: u64,
    pub privacy_level_distribution: std::collections::HashMap<String, u64>,
}

impl Default for QuantumPrivacyStats {
    fn default() -> Self {
        Self {
            total_private_trades: 0,
            zk_proof_success_rate_bps: 10000, // 100%
            tor_circuits_used: 0,
            privacy_level_distribution: std::collections::HashMap::new(),
        }
    }
}

/// OHLCV data with quantum security
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumOhlcvData {
    pub timestamp: DateTime<Utc>,
    pub open: BigDecimal,
    pub high: BigDecimal,
    pub low: BigDecimal,
    pub close: BigDecimal,
    pub volume: BigDecimal,
    pub quantum_hash: Option<Vec<u8>>,
}

/// DexScreener compatible response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexScreenerResponse {
    pub pairs: Vec<DexScreenerPair>,
    pub schema_version: String,
    pub generated_at: DateTime<Utc>,
    pub quantum_secured: bool,
}

/// DexScreener pair format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexScreenerPair {
    pub chain_id: String,
    pub dex_id: String,
    pub pair_address: String,
    pub base_token: DexScreenerToken,
    pub quote_token: DexScreenerToken,
    pub price_native: String,
    pub price_usd: String,
    pub liquidity: DexScreenerLiquidity,
    pub fdv: Option<String>,
    pub volume: DexScreenerVolume,
    pub info: DexScreenerInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexScreenerToken {
    pub address: String,
    pub name: String,
    pub symbol: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexScreenerLiquidity {
    pub usd: String,
    pub base: String,
    pub quote: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexScreenerVolume {
    pub h24: String,
    pub h6: String,
    pub h1: String,
    pub m5: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexScreenerInfo {
    pub image_url: Option<String>,
    pub websites: Vec<DexScreenerWebsite>,
    pub socials: Vec<DexScreenerSocial>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexScreenerWebsite {
    pub label: String,
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexScreenerSocial {
    #[serde(rename = "type")]
    pub social_type: String,
    pub url: String,
}
