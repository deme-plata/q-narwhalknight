//! Q-DEX - Quantum-Enhanced Decentralized Exchange
//!
//! Quantum-enhanced DEX with physics-inspired trading algorithms,
//! post-quantum security, and integration with Q-NarwhalKnight consensus.
//!
//! Features:
//! - Quantum-resistant cryptographic signatures
//! - Physics-inspired price discovery using quantum field theory
//! - Heisenberg uncertainty-based volatility modeling
//! - Quantum entangled liquidity pools
//! - ZK-SNARK privacy for transactions
//! - Integration with Q-Oracle for quantum random pricing
//! - Native ORBUSD stablecoin support

use anyhow::Result;
use bigdecimal::{BigDecimal, ToPrimitive};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

pub mod analytics;
pub mod api;
pub mod liquidity;
pub mod oracle_price_bridge;
pub mod screener;
pub mod trading;
pub mod types;

// Import from submodules
use self::analytics::QuantumTradingAnalytics;
use self::api::QuantumDexApiServer;
use self::liquidity::QuantumLiquidityManager;
use self::screener::QuantumDexScreenerIntegration;
use self::trading::QuantumTradingEngine;
use q_storage::token_registry::{TokenRegistry, TokenMetadata, PoolMetadata};
use q_storage::price_history::{PriceHistoryManager, TradeRecord};

pub use analytics::*;
pub use api::*;
pub use liquidity::*;
pub use oracle_price_bridge::*;
pub use screener::*;
pub use trading::*;
pub use types::*;

/// Quantum-Enhanced DEX Integration Manager
/// Main coordinator for quantum-secure exchange operations
#[derive(Clone)]
pub struct QuantumDexManager {
    pub api_server: Arc<QuantumDexApiServer>,
    pub screener: Arc<QuantumDexScreenerIntegration>,
    pub liquidity: Arc<QuantumLiquidityManager>,
    pub trading: Arc<QuantumTradingEngine>,
    pub analytics: Arc<QuantumTradingAnalytics>,

    // NEW: Persistent storage systems
    pub token_registry: Arc<TokenRegistry>,
    pub price_history: Arc<PriceHistoryManager>,

    // Quantum-enhanced data stores (now mostly cached from registry)
    pub token_data: Arc<RwLock<HashMap<String, QuantumTokenInfo>>>,
    pub pair_data: Arc<RwLock<HashMap<String, QuantumTradingPair>>>,
    pub market_data: Arc<RwLock<QuantumMarketData>>,
    pub price_feeds: Arc<RwLock<HashMap<String, crate::types::QuantumPriceFeed>>>,
    pub quantum_params: Arc<RwLock<QuantumDexParameters>>,
}

impl QuantumDexManager {
    /// Create a new quantum-enhanced DEX manager
    pub fn new(token_registry: Arc<TokenRegistry>, price_history: Arc<PriceHistoryManager>) -> Result<Self> {
        // Liquidity manager is constructed first and shared with the trading engine
        // so that execute_atomic_swap() can update pool reserves (DEX-001/002).
        let liquidity = Arc::new(QuantumLiquidityManager::new());
        Ok(Self {
            api_server: Arc::new(QuantumDexApiServer::new(8080)),
            screener: Arc::new(QuantumDexScreenerIntegration::new()),
            trading: Arc::new(QuantumTradingEngine::new(Arc::clone(&liquidity))),
            liquidity,
            analytics: Arc::new(QuantumTradingAnalytics::new()),
            token_registry,
            price_history,
            token_data: Arc::new(RwLock::new(HashMap::new())),
            pair_data: Arc::new(RwLock::new(HashMap::new())),
            market_data: Arc::new(RwLock::new(QuantumMarketData::default())),
            price_feeds: Arc::new(RwLock::new(HashMap::new())),
            quantum_params: Arc::new(RwLock::new(QuantumDexParameters::default())),
        })
    }

    /// Initialize the quantum DEX system
    pub async fn initialize(&self) -> Result<()> {
        info!("🚀 Initializing Quantum-Enhanced DEX System");
        info!("⚛️ Quantum physics-inspired trading algorithms activated");

        // Setup quantum parameters
        self.setup_quantum_parameters().await?;

        // Initialize quantum-enhanced tokens
        self.setup_quantum_tokens().await?;

        // v1.1.23-beta: DISABLED - DEX API is now integrated into main server at /api/dex
        // Starting a separate server on port 8080 caused port conflicts:
        // - DEX server would grab port 8080 first
        // - Main HTTP server (with mining routes) would fall back to 8082
        // - Users mapping port 8080 in Docker would hit DEX, not main server
        // The DEX routes are already integrated via .nest("/api/dex", ...) in main.rs
        info!("🚀 Quantum DEX routes integrated into main HTTP server (no separate server)");

        // Initialize quantum DexScreener integration
        self.screener.initialize().await?;

        // Start quantum liquidity tracking
        self.liquidity.start_quantum_tracking().await?;

        // Initialize quantum trading engine
        self.trading.initialize_quantum_engine().await?;

        // Start quantum analytics collection
        self.analytics.start_quantum_collection().await?;

        // Start quantum data updates
        self.start_quantum_data_updates().await?;

        info!("✅ Quantum-Enhanced DEX System initialized successfully");
        info!("🎯 Physics-inspired algorithms: ACTIVE");
        info!("🔒 Post-quantum cryptography: ENABLED");
        info!("⚡ Native ORBUSD integration: READY");

        Ok(())
    }

    /// Setup quantum physics parameters for the DEX
    async fn setup_quantum_parameters(&self) -> Result<()> {
        let mut params = self.quantum_params.write().await;

        // Physics constants scaled for financial applications
        *params = QuantumDexParameters {
            planck_constant: "0.00000000000000000000000000000000066260701".parse().unwrap(), // For volatility scaling
            golden_ratio: "1.618033988749895".parse().unwrap(), // For price discovery
            euler_constant: "2.718281828459045".parse().unwrap(), // For liquidity curves
            pi_constant: "3.141592653589793".parse().unwrap(),  // For wave functions

            // Quantum-specific trading parameters
            uncertainty_principle_factor: 0.1618, // Golden ratio scaled
            wave_collapse_threshold: 0.05,        // 5% price movement triggers wave collapse
            entanglement_strength: 0.707,         // √2/2 for quantum correlation
            decoherence_time_seconds: 300,        // 5 minutes quantum state lifetime

            // Risk management parameters
            max_leverage: 10.0,
            liquidation_threshold_bps: 8000, // 80%
            slippage_protection_bps: 50,     // 0.5% max slippage
        };

        info!("⚛️ Quantum physics parameters configured");
        Ok(())
    }

    /// Setup quantum-enhanced token data - now loads from registry
    async fn setup_quantum_tokens(&self) -> Result<()> {
        info!("💎 Loading quantum token data from registry");

        // Bootstrap default tokens if registry is empty
        self.bootstrap_default_tokens().await?;

        // Load all tokens from registry into cache for fast access
        let tokens = self.token_registry.get_all_tokens().await?;

        let mut token_data = self.token_data.write().await;
        for token in tokens {
            // Convert TokenMetadata to QuantumTokenInfo
            let quantum_token = self.convert_to_quantum_token_info(&token);
            token_data.insert(token.symbol.clone(), quantum_token);
        }

        // Load all pools from registry
        let pools = self.token_registry.get_all_pools().await?;

        let mut pair_data = self.pair_data.write().await;
        for pool in pools {
            // Convert PoolMetadata to QuantumTradingPair
            let quantum_pair = self.convert_to_quantum_trading_pair(&pool);
            pair_data.insert(pool.pair_id.clone(), quantum_pair);
        }

        info!("✅ Loaded {} tokens and {} pools from registry", token_data.len(), pair_data.len());
        Ok(())
    }

    /// Bootstrap default tokens (ORB and ORBUSD) if registry is empty
    async fn bootstrap_default_tokens(&self) -> Result<()> {
        // Check if ORB token already exists
        if let Ok(Some(_)) = self.token_registry.get_token_by_symbol("ORB").await {
            return Ok(()); // Already bootstrapped
        }

        info!("🌱 Bootstrapping default tokens (ORB and ORBUSD)");

        // Register ORB token
        let orb_token = TokenMetadata {
            contract_address: "0x0000000000000000000000000000000000000ORB".to_string(),
            symbol: "ORB".to_string(),
            name: "OroBit Quantum Token".to_string(),
            decimals: 18,
            total_supply: BigDecimal::from(21_000_000),
            circulating_supply: BigDecimal::from(0),
            creator: "system".to_string(),
            created_at: Utc::now(),
            is_verified: true,
            is_active: true,
            price_usd: BigDecimal::from(0),
            market_cap: BigDecimal::from(0),
            volume_24h: BigDecimal::from(0),
            price_change_24h_bps: 0,
            logo_url: Some("https://q-narwhalknight.xyz/orb-logo.png".to_string()),
            website: Some("https://q-narwhalknight.xyz".to_string()),
            description: Some("ORB - Quantum-enhanced governance token for Q-NarwhalKnight with post-quantum security".to_string()),
            tags: vec!["quantum".to_string(), "governance".to_string(), "defi".to_string(), "post-quantum".to_string()],
            has_liquidity_pool: false,
            liquidity_pools: vec![],
            last_updated: Utc::now(),
            social_links: None, // v2.7.7-beta
        };
        self.token_registry.register_token(orb_token).await?;

        // Register ORBUSD token
        let orbusd_token = TokenMetadata {
            contract_address: "0x0000000000000000000000000000000ORBUSD".to_string(),
            symbol: "ORBUSD".to_string(),
            name: "OroBit USD Quantum Stablecoin".to_string(),
            decimals: 18,
            total_supply: BigDecimal::from(0), // Algorithmic supply
            circulating_supply: BigDecimal::from(0),
            creator: "system".to_string(),
            created_at: Utc::now(),
            is_verified: true,
            is_active: true,
            price_usd: BigDecimal::from(1), // Quantum-stabilized at $1
            market_cap: BigDecimal::from(0),
            volume_24h: BigDecimal::from(0),
            price_change_24h_bps: 0,
            logo_url: Some("https://q-narwhalknight.xyz/orbusd-logo.png".to_string()),
            website: Some("https://q-narwhalknight.xyz/orbusd".to_string()),
            description: Some("ORBUSD - Physics-inspired algorithmic stablecoin with quantum uncertainty-based stability".to_string()),
            tags: vec!["stablecoin".to_string(), "algorithmic".to_string(), "quantum".to_string(), "physics".to_string()],
            has_liquidity_pool: false,
            liquidity_pools: vec![],
            last_updated: Utc::now(),
            social_links: None, // v2.7.7-beta
        };
        self.token_registry.register_token(orbusd_token).await?;

        info!("✅ Default tokens bootstrapped");
        Ok(())
    }

    /// Convert TokenMetadata to QuantumTokenInfo
    fn convert_to_quantum_token_info(&self, token: &TokenMetadata) -> QuantumTokenInfo {
        QuantumTokenInfo {
            symbol: token.symbol.clone(),
            name: token.name.clone(),
            decimals: token.decimals,
            contract_address: Some(token.contract_address.clone()),
            total_supply: token.total_supply.clone(),
            quantum_secured: token.is_verified,
            privacy_enabled: true,
            zk_proofs_required: false,
            created_at: token.created_at,

            // Market data fields (Option types)
            price_usd: Some(token.price_usd.clone()),
            market_cap: Some(token.market_cap.clone()),
            circulating_supply: Some(token.circulating_supply.clone()),
            volume_24h: Some(token.volume_24h.clone()),

            // Token metadata
            description: token.description.clone(),
            logo_url: token.logo_url.clone(),
            website: token.website.clone(),
            tags: token.tags.clone(),
            address: Some(token.contract_address.clone()),
            quantum_signature_verified: token.is_verified,

            // Quantum-specific properties
            quantum_volatility: "0.1618".parse().unwrap(),
            wave_function_state: QuantumState::Superposition,
            entanglement_pairs: vec![],
            defi_protocols: vec!["QuantumDEX".to_string()],
        }
    }

    /// Convert PoolMetadata to QuantumTradingPair
    fn convert_to_quantum_trading_pair(&self, pool: &PoolMetadata) -> QuantumTradingPair {
        // Calculate current price from reserves
        let price = if pool.reserve_quote > BigDecimal::from(0) {
            &pool.reserve_base / &pool.reserve_quote
        } else {
            BigDecimal::from(0)
        };

        QuantumTradingPair {
            id: pool.pair_id.clone(),
            pair_id: pool.pair_id.clone(),
            base_token: pool.base_token.clone(),
            quote_token: pool.quote_token.clone(),
            base_address: Some(pool.base_token_address.clone()),
            quote_address: Some(pool.quote_token_address.clone()),
            exchange: "QuantumDEX".to_string(),
            price,
            volume_24h: pool.volume_24h.clone(),
            liquidity: pool.liquidity_usd.clone(),
            // Convert fee_rate to basis points using integer math (no f64 precision loss)
            // fee_rate is a BigDecimal like 0.003 (0.3%), multiply by 10000 to get 30 basis points
            fee_rate: (&pool.fee_rate * BigDecimal::from(10000))
                .to_u16()
                .unwrap_or(30), // 30 basis points = 0.3% default
            fee_tier: pool.fee_rate.clone(),
            min_trade_size: "0.0001".parse().unwrap(), // v8.6.0: lowered from 0.001 for smaller trades
            max_trade_size: "10000000".parse().unwrap(), // v8.6.0: raised from 1M to 10M for larger trades
            quantum_secured: true,
            privacy_tier: QuantumPrivacyTier::Quantum,
            zk_proof_required: false,
            created_at: pool.created_at,
            active: pool.is_active,

            // Quantum pair properties
            quantum_correlation: 0.707,
            wave_interference_pattern: WavePattern::Constructive,
            price_uncertainty: "0.01".parse().unwrap(),
            quantum_liquidity_depth: pool.liquidity_usd.clone(),
            entangled_state: true,
        }
    }

    /// Start quantum-enhanced data updates
    async fn start_quantum_data_updates(&self) -> Result<()> {
        let analytics = self.analytics.clone();
        let market_data = self.market_data.clone();
        let price_feeds = self.price_feeds.clone();
        let quantum_params = self.quantum_params.clone();

        // Quantum price update task with uncertainty principle
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(5)); // High frequency quantum updates

            loop {
                interval.tick().await;

                // Apply quantum price discovery with uncertainty
                if let Ok(quantum_price) = analytics.collect_quantum_price_data("ORB/ORBUSD").await
                {
                    let mut feeds = price_feeds.write().await;

                    // Add quantum uncertainty to price
                    use std::str::FromStr;
                    let params = quantum_params.read().await;
                    let uncertainty_factor = params.uncertainty_principle_factor;
                    let multiplier = BigDecimal::from_str(&(1.0 + uncertainty_factor).to_string())
                        .unwrap_or_else(|_| "1.01618".parse().unwrap());
                    let price_with_uncertainty = quantum_price * multiplier;

                    feeds.insert(
                        "ORB/ORBUSD".to_string(),
                        crate::types::QuantumPriceFeed {
                            symbol: "ORB/ORBUSD".to_string(),
                            price: price_with_uncertainty,
                            timestamp: Utc::now(),
                            source: "QuantumDEX".to_string(),
                            quantum_uncertainty: BigDecimal::from_str(&uncertainty_factor.to_string())
                                .unwrap_or_else(|_| "0.01618".parse().unwrap()),
                            wave_function_collapsed: true,
                            entanglement_strength: 0.707,
                        },
                    );
                }
            }
        });

        // Quantum market data with wave function analysis
        let analytics_clone = self.analytics.clone();
        let market_data_clone = self.market_data.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));

            loop {
                interval.tick().await;

                if let Ok(quantum_data) = analytics_clone.collect_quantum_market_data().await {
                    *market_data_clone.write().await = quantum_data;
                }
            }
        });

        info!("⚡ Quantum data update loops started with physics-based algorithms");
        Ok(())
    }

    /// Execute quantum-enhanced trade with post-quantum security
    pub async fn execute_quantum_trade(
        &self,
        trade_request: QuantumTradeRequest,
    ) -> Result<QuantumTradeResult> {
        // Physics-layer pricing (display/UX purpose)
        let mut result = self.trading.execute_quantum_trade(&trade_request).await?;

        // DEX-001/002: atomically update pool reserves using constant-product AMM.
        // Physics price above is for display; settled amount comes from the AMM formula.
        // DEX-003: derive slippage floor from request.max_slippage_bps + physics price.
        let min_out = if trade_request.max_slippage_bps > 0 {
            let expected_out = &trade_request.amount * &result.price;
            let scale_num = BigDecimal::from(10_000i64 - trade_request.max_slippage_bps as i64);
            expected_out * scale_num / BigDecimal::from(10_000i64)
        } else {
            BigDecimal::from(0i64)
        };

        let (actual_out, _, _) = self.liquidity
            .execute_atomic_swap(&trade_request.pair_id, &trade_request.amount, &min_out)
            .await?;

        // Use AMM-settled output as the authoritative filled amount
        result.amount_filled = actual_out;

        Ok(result)
    }

    /// Add quantum-entangled liquidity to a pair
    pub async fn add_quantum_liquidity(
        &self,
        pair_id: &str,
        amount_a: BigDecimal,
        amount_b: BigDecimal,
        provider: &str,
    ) -> Result<QuantumLiquidityPosition> {
        let quantum_request = QuantumTradeRequest {
            user: provider.to_string(),
            trader_id: provider.to_string(),
            pair_id: pair_id.to_string(),
            side: TradeSide::Buy, // Liquidity provision treated as buy side
            amount: amount_a.clone(),
            price: None,
            order_type: OrderType::Market,
            privacy_level: QuantumPrivacyTier::Basic,
            zk_proof_required: false,
            max_slippage_bps: 100, // 1% default slippage
            expires_at: None,
            quantum_signature: vec![0u8; 64], // Post-quantum signature placeholder
            entanglement_proof: Some(vec![0u8; 32]),
        };
        self.liquidity.add_quantum_liquidity(&quantum_request).await
    }

    /// Get quantum trading analytics with wave function analysis
    pub async fn get_quantum_trading_analytics(
        &self,
        timeframe: QuantumTimeframe,
    ) -> Result<QuantumTradingStats> {
        self.analytics.get_quantum_trading_stats(&timeframe).await
    }

    /// Generate DexScreener data with quantum enhancements
    pub async fn get_quantum_dexscreener_data(&self) -> Result<QuantumDexScreenerResponse> {
        self.screener.generate_quantum_response().await
    }

    /// Get quantum OHLCV data with wave interference patterns
    pub async fn get_quantum_ohlcv_data(
        &self,
        pair_id: &str,
        timeframe: &str,
        limit: Option<u32>,
    ) -> Result<Vec<QuantumOhlcvData>> {
        let quantum_timeframe = match timeframe {
            "1m" => QuantumTimeframe::Minute1,
            "5m" => QuantumTimeframe::Minute5,
            "15m" => QuantumTimeframe::Minute15,
            "1h" => QuantumTimeframe::Hour1,
            "4h" => QuantumTimeframe::Hour4,
            "1d" => QuantumTimeframe::Day1,
            "1w" => QuantumTimeframe::Week1,
            _ => QuantumTimeframe::Day1,
        };
        self.analytics
            .get_quantum_ohlcv(pair_id, &quantum_timeframe)
            .await
    }

    /// Get token information with quantum properties
    pub async fn get_quantum_token_info(&self, symbol: &str) -> Result<QuantumTokenInfo> {
        self.token_data
            .read()
            .await
            .get(symbol)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Quantum token not found: {}", symbol))
    }

    /// Get quantum market data with physics analysis
    pub async fn get_quantum_market_data(&self) -> QuantumMarketData {
        self.market_data.read().await.clone()
    }

    /// Update quantum price with wave function collapse
    pub async fn update_quantum_price(
        &self,
        symbol: &str,
        price: BigDecimal,
        collapsed: bool,
    ) -> Result<()> {
        let mut token_data = self.token_data.write().await;
        if let Some(token) = token_data.get_mut(symbol) {
            token.price_usd = Some(price.clone());
            // Calculate market cap: circulating_supply * price_usd
            if let (Some(circ_supply), Some(price_val)) = (&token.circulating_supply, &token.price_usd) {
                token.market_cap = Some(circ_supply * price_val);
            }
            token.wave_function_state = if collapsed {
                QuantumState::Collapsed
            } else {
                QuantumState::Superposition
            };
        }
        Ok(())
    }

    // ============ NEW: TOKEN REGISTRY INTEGRATION ============

    /// Register a new token (called from VM when token is created)
    pub async fn register_token_from_vm(
        &self,
        contract_address: String,
        symbol: String,
        name: String,
        decimals: u8,
        total_supply: BigDecimal,
        creator: String,
    ) -> Result<()> {
        info!("🪙 Registering new token from VM: {} ({})", symbol, contract_address);

        let token = TokenMetadata {
            contract_address: contract_address.clone(),
            symbol: symbol.clone(),
            name,
            decimals,
            total_supply: total_supply.clone(),
            circulating_supply: total_supply, // Initially all circulating
            creator,
            created_at: Utc::now(),
            is_verified: false, // New tokens start unverified
            is_active: true,
            price_usd: BigDecimal::from(0),
            market_cap: BigDecimal::from(0),
            volume_24h: BigDecimal::from(0),
            price_change_24h_bps: 0,
            logo_url: None,
            website: None,
            description: None,
            tags: vec!["custom".to_string()],
            has_liquidity_pool: false,
            liquidity_pools: vec![],
            last_updated: Utc::now(),
            social_links: None, // v2.7.7-beta
        };

        // Register in persistent storage
        self.token_registry.register_token(token.clone()).await?;

        // Update in-memory cache
        let quantum_token = self.convert_to_quantum_token_info(&token);
        self.token_data.write().await.insert(symbol, quantum_token);

        info!("✅ Token registered and available in DEX");
        Ok(())
    }

    /// Register a new liquidity pool (called when pool is created)
    pub async fn register_liquidity_pool(
        &self,
        pool_address: String,
        base_token: String,
        quote_token: String,
        initial_reserve_base: BigDecimal,
        initial_reserve_quote: BigDecimal,
        creator: String,
    ) -> Result<String> {
        info!("🏊 Registering new liquidity pool: {}/{}", base_token, quote_token);

        // Get token addresses
        let base_token_meta = self.token_registry.get_token_by_symbol(&base_token).await?
            .ok_or_else(|| anyhow::anyhow!("Base token not found: {}", base_token))?;
        let quote_token_meta = self.token_registry.get_token_by_symbol(&quote_token).await?
            .ok_or_else(|| anyhow::anyhow!("Quote token not found: {}", quote_token))?;

        let pair_id = format!("{}/{}", base_token, quote_token);
        let pool_id = format!("pool-{}-{}", base_token, quote_token).to_lowercase();

        // Calculate initial shares (geometric mean)
        let initial_shares = (&initial_reserve_base * &initial_reserve_quote)
            .sqrt()
            .ok_or_else(|| anyhow::anyhow!("Cannot calculate initial shares"))?;

        let pool = PoolMetadata {
            pool_id: pool_id.clone(),
            pool_address: pool_address.clone(),
            pair_id: pair_id.clone(),
            base_token: base_token.clone(),
            quote_token: quote_token.clone(),
            base_token_address: base_token_meta.contract_address,
            quote_token_address: quote_token_meta.contract_address,
            reserve_base: initial_reserve_base,
            reserve_quote: initial_reserve_quote,
            total_shares: initial_shares,
            fee_rate: "0.010".parse().unwrap(), // v8.6.0: 1.0% protocol fee (was 0.3%)
            created_at: Utc::now(),
            creator,
            is_active: true,
            is_paused: false,
            liquidity_locked_until: None,
            volume_24h: BigDecimal::from(0),
            fees_24h: BigDecimal::from(0),
            liquidity_usd: BigDecimal::from(0), // Will be calculated
            apr: 0.0,
            provider_count: 1, // Creator is first provider
            last_updated: Utc::now(),
        };

        // Register in persistent storage
        self.token_registry.register_pool(pool.clone()).await?;

        // Update in-memory cache
        let quantum_pair = self.convert_to_quantum_trading_pair(&pool);
        self.pair_data.write().await.insert(pair_id, quantum_pair);

        info!("✅ Liquidity pool registered: {}", pool_id);
        Ok(pool_id)
    }

    /// Record a trade (updates price history and registry)
    pub async fn record_trade(
        &self,
        trade_result: &QuantumTradeResult,
    ) -> Result<()> {
        use q_storage::price_history::{TradeRecord, TradeSide as StorageTradeSide};

        // Convert to storage trade record
        let trade_record = TradeRecord {
            trade_id: trade_result.trade_id.clone(),
            pair_id: trade_result.pair_id.clone(),
            timestamp: trade_result.executed_at,
            price: trade_result.price.clone(),
            amount: trade_result.amount_filled.clone(),
            side: match trade_result.side {
                TradeSide::Buy => StorageTradeSide::Buy,
                TradeSide::Sell => StorageTradeSide::Sell,
            },
            trader: Some(trade_result.trader_id.clone()),
        };

        // Record in price history (this updates OHLCV candles)
        self.price_history.record_trade(trade_record).await?;

        // Update token registry with latest price and volume
        let tokens: Vec<String> = trade_result.pair_id.split('/').map(|s| s.to_string()).collect();
        if tokens.len() == 2 {
            if let Ok(Some(base_token)) = self.token_registry.get_token_by_symbol(&tokens[0]).await {
                self.token_registry.update_token_price(
                    &base_token.contract_address,
                    trade_result.price.clone(),
                    trade_result.amount_filled.clone(),
                ).await?;
            }
        }

        debug!("📊 Trade recorded in price history: {}", trade_result.trade_id);
        Ok(())
    }

    /// Get all available tokens from registry (replaces hardcoded list)
    pub async fn get_all_available_tokens(&self) -> Result<Vec<QuantumTokenInfo>> {
        let tokens = self.token_registry.get_active_tokens().await?;
        Ok(tokens.iter().map(|t| self.convert_to_quantum_token_info(t)).collect())
    }

    /// Get historical price data from price history manager
    pub async fn get_historical_prices(
        &self,
        pair_id: &str,
        interval: &str,
        limit: Option<usize>,
    ) -> Result<Vec<QuantumOhlcvData>> {
        use q_storage::price_history::CandleInterval;

        let candle_interval = CandleInterval::from_str(interval)
            .unwrap_or(CandleInterval::Hour1);

        let candles = self.price_history.get_recent_candles(pair_id, candle_interval, limit.unwrap_or(100)).await?;

        // Convert to QuantumOhlcvData
        Ok(candles.iter().map(|c| QuantumOhlcvData {
            timestamp: c.timestamp,
            open: c.open.clone(),
            high: c.high.clone(),
            low: c.low.clone(),
            close: c.close.clone(),
            volume: c.volume.clone(),
            quantum_hash: Some(vec![0u8; 32]), // Placeholder
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use rocksdb::DB;
    use q_storage::price_history::PriceHistoryManager;

    async fn create_test_dex_manager() -> QuantumDexManager {
        let temp_dir = tempdir().unwrap();
        let db = Arc::new(DB::open_default(temp_dir.path()).unwrap());

        // Create token registry with test database
        let token_registry = Arc::new(TokenRegistry::new(db.clone()));
        token_registry.initialize().await.unwrap();

        // Create price history manager with same database
        let price_history = Arc::new(PriceHistoryManager::new(db));
        price_history.initialize().await.unwrap();

        // Keep temp_dir alive by leaking it (for test purposes only)
        std::mem::forget(temp_dir);

        QuantumDexManager::new(token_registry, price_history).unwrap()
    }

    #[tokio::test]
    async fn test_quantum_dex_creation() {
        let manager = create_test_dex_manager().await;
        assert!(manager.initialize().await.is_ok());
    }

    #[tokio::test]
    async fn test_quantum_token_retrieval() {
        let manager = create_test_dex_manager().await;
        manager.initialize().await.unwrap();

        let orb_info = manager.get_quantum_token_info("ORB").await.unwrap();
        assert_eq!(orb_info.symbol, "ORB");
        assert_eq!(orb_info.name, "OroBit Quantum Token");
        assert!(orb_info.quantum_signature_verified);
    }

    #[tokio::test]
    async fn test_quantum_physics_parameters() {
        let manager = create_test_dex_manager().await;
        manager.initialize().await.unwrap();

        let params = manager.quantum_params.read().await;
        assert_eq!(params.golden_ratio, "1.618033988749895".parse().unwrap());
        assert_eq!(params.uncertainty_principle_factor, 0.1618);
    }
}
