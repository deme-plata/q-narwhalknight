//! Token Registry Storage System
//!
//! Persistent storage for all tokens created in the Q-NarwhalKnight ecosystem.
//! This module provides a complete registry of tokens including their metadata,
//! liquidity pools, and trading pairs.

use anyhow::Result;
use bigdecimal::BigDecimal;
use chrono::{DateTime, Utc};
#[cfg(not(target_os = "windows"))]
use rocksdb::{IteratorMode, DB};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// v2.7.7-beta: Social media links for tokens
/// Supports common platforms: Twitter/X, Discord, Telegram, Website, GitHub
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SocialLinks {
    /// Twitter/X handle (e.g., "https://x.com/ViktorakaDeme" or "@handle")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub twitter: Option<String>,
    /// Discord server invite link
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discord: Option<String>,
    /// Telegram group/channel link
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telegram: Option<String>,
    /// Official website URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub website: Option<String>,
    /// GitHub repository URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub github: Option<String>,
    /// Medium blog URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub medium: Option<String>,
    /// Reddit community URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reddit: Option<String>,
    /// CoinMarketCap listing URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coinmarketcap: Option<String>,
    /// CoinGecko listing URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coingecko: Option<String>,
}

/// Token metadata stored in registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenMetadata {
    pub contract_address: String,
    pub symbol: String,
    pub name: String,
    pub decimals: u8,
    pub total_supply: BigDecimal,
    pub circulating_supply: BigDecimal,
    pub creator: String,
    pub created_at: DateTime<Utc>,
    pub is_verified: bool,
    pub is_active: bool,

    // Market data (updated dynamically)
    pub price_usd: BigDecimal,
    pub market_cap: BigDecimal,
    pub volume_24h: BigDecimal,
    /// Price change in 24h in basis points (e.g., 550 = 5.5%, -200 = -2%)
    pub price_change_24h_bps: i32,

    // Metadata
    pub logo_url: Option<String>,
    pub website: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,

    // v2.7.7-beta: Social media links
    #[serde(default)]
    pub social_links: Option<SocialLinks>,

    // DEX integration
    pub has_liquidity_pool: bool,
    pub liquidity_pools: Vec<String>, // Pool IDs this token is part of

    // Updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Liquidity pool metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolMetadata {
    pub pool_id: String,
    pub pool_address: String,
    pub pair_id: String, // e.g., "ORB/MYTOKEN"
    pub base_token: String,
    pub quote_token: String,
    pub base_token_address: String,
    pub quote_token_address: String,

    // Reserves
    pub reserve_base: BigDecimal,
    pub reserve_quote: BigDecimal,
    pub total_shares: BigDecimal,

    // Pool parameters
    pub fee_rate: BigDecimal, // e.g., 0.003 for 0.3%
    pub created_at: DateTime<Utc>,
    pub creator: String,

    // Pool state
    pub is_active: bool,
    pub is_paused: bool,
    pub liquidity_locked_until: Option<DateTime<Utc>>,

    // Market metrics
    pub volume_24h: BigDecimal,
    pub fees_24h: BigDecimal,
    pub liquidity_usd: BigDecimal,
    pub apr: f64, // Annual percentage rate

    // Providers
    pub provider_count: u64,

    // Updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Trading pair metadata (for tracking active trading pairs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingPairMetadata {
    pub pair_id: String,
    pub base_token: String,
    pub quote_token: String,
    pub pool_address: Option<String>,
    pub current_price: BigDecimal,
    pub volume_24h: BigDecimal,
    pub high_24h: BigDecimal,
    pub low_24h: BigDecimal,
    pub trades_count_24h: u64,
    pub last_trade_timestamp: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
}

/// Token Registry - Central storage for all token-related data
#[cfg(not(target_os = "windows"))]
pub struct TokenRegistry {
    db: Arc<DB>,

    // In-memory caches for fast access
    token_cache: Arc<RwLock<HashMap<String, TokenMetadata>>>,
    pool_cache: Arc<RwLock<HashMap<String, PoolMetadata>>>,
    pair_cache: Arc<RwLock<HashMap<String, TradingPairMetadata>>>,

    // Index mappings for efficient lookups
    symbol_to_address: Arc<RwLock<HashMap<String, String>>>,
    address_to_pools: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

#[cfg(target_os = "windows")]
pub struct TokenRegistry {
    // In-memory caches for fast access
    token_cache: Arc<RwLock<HashMap<String, TokenMetadata>>>,
    pool_cache: Arc<RwLock<HashMap<String, PoolMetadata>>>,
    pair_cache: Arc<RwLock<HashMap<String, TradingPairMetadata>>>,

    // Index mappings for efficient lookups
    symbol_to_address: Arc<RwLock<HashMap<String, String>>>,
    address_to_pools: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

#[cfg(not(target_os = "windows"))]
impl TokenRegistry {
    /// Create a new token registry with RocksDB backend
    pub fn new(db: Arc<DB>) -> Self {
        Self {
            db,
            token_cache: Arc::new(RwLock::new(HashMap::new())),
            pool_cache: Arc::new(RwLock::new(HashMap::new())),
            pair_cache: Arc::new(RwLock::new(HashMap::new())),
            symbol_to_address: Arc::new(RwLock::new(HashMap::new())),
            address_to_pools: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Initialize the registry and load data from disk
    pub async fn initialize(&self) -> Result<()> {
        info!("🗄️  Initializing Token Registry");

        // Load all tokens from database
        self.load_tokens_from_db().await?;

        // Load all pools from database
        self.load_pools_from_db().await?;

        // Load all trading pairs from database
        self.load_pairs_from_db().await?;

        // Build index mappings
        self.rebuild_indices().await?;

        let token_count = self.token_cache.read().await.len();
        let pool_count = self.pool_cache.read().await.len();

        info!("✅ Token Registry initialized: {} tokens, {} pools", token_count, pool_count);
        Ok(())
    }

    // ============ TOKEN OPERATIONS ============

    /// Register a new token in the registry
    pub async fn register_token(&self, token: TokenMetadata) -> Result<()> {
        info!("📝 Registering token: {} ({})", token.symbol, token.contract_address);

        // Store in database
        let key = format!("token:{}", token.contract_address);
        let value = bincode::serialize(&token)?;
        self.db.put(key.as_bytes(), value)?;

        // Update cache
        self.token_cache.write().await.insert(token.contract_address.clone(), token.clone());

        // Update symbol index
        self.symbol_to_address.write().await.insert(token.symbol.clone(), token.contract_address.clone());

        info!("✅ Token registered: {}", token.symbol);
        Ok(())
    }

    /// Get token by address
    pub async fn get_token_by_address(&self, address: &str) -> Result<Option<TokenMetadata>> {
        // Check cache first
        if let Some(token) = self.token_cache.read().await.get(address) {
            return Ok(Some(token.clone()));
        }

        // Load from database
        let key = format!("token:{}", address);
        if let Some(value) = self.db.get(key.as_bytes())? {
            let token: TokenMetadata = bincode::deserialize(&value)?;

            // Update cache
            self.token_cache.write().await.insert(address.to_string(), token.clone());

            Ok(Some(token))
        } else {
            Ok(None)
        }
    }

    /// Get token by symbol
    pub async fn get_token_by_symbol(&self, symbol: &str) -> Result<Option<TokenMetadata>> {
        if let Some(address) = self.symbol_to_address.read().await.get(symbol) {
            self.get_token_by_address(address).await
        } else {
            Ok(None)
        }
    }

    /// Get all registered tokens
    pub async fn get_all_tokens(&self) -> Result<Vec<TokenMetadata>> {
        let cache = self.token_cache.read().await;
        let mut tokens: Vec<TokenMetadata> = cache.values().cloned().collect();

        // Sort by creation date (newest first)
        tokens.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(tokens)
    }

    /// Get all active tokens (excluding inactive/delisted)
    pub async fn get_active_tokens(&self) -> Result<Vec<TokenMetadata>> {
        let all_tokens = self.get_all_tokens().await?;
        Ok(all_tokens.into_iter().filter(|t| t.is_active).collect())
    }

    /// Update token metadata (price, volume, etc.)
    pub async fn update_token_metadata(&self, address: &str, update_fn: impl FnOnce(&mut TokenMetadata)) -> Result<()> {
        if let Some(mut token) = self.get_token_by_address(address).await? {
            update_fn(&mut token);
            token.last_updated = Utc::now();
            self.register_token(token).await?;
        }
        Ok(())
    }

    /// Update token price
    pub async fn update_token_price(&self, address: &str, price_usd: BigDecimal, volume_24h: BigDecimal) -> Result<()> {
        self.update_token_metadata(address, |token| {
            let old_price = token.price_usd.clone();
            token.price_usd = price_usd.clone();
            token.volume_24h = volume_24h;

            // Calculate 24h price change in basis points
            // Formula: ((new - old) / old) * 10000 = basis points
            if old_price > BigDecimal::from(0) {
                let change = &price_usd - &old_price;
                let change_ratio = &change / &old_price;
                // Convert to basis points: multiply by 10000
                let bps_decimal = change_ratio * BigDecimal::from(10000);
                token.price_change_24h_bps = bps_decimal.to_string().parse::<f64>().unwrap_or(0.0) as i32;
            }

            // Update market cap
            token.market_cap = &token.circulating_supply * &token.price_usd;
        }).await
    }

    // ============ LIQUIDITY POOL OPERATIONS ============

    /// Register a new liquidity pool
    pub async fn register_pool(&self, pool: PoolMetadata) -> Result<()> {
        info!("🏊 Registering liquidity pool: {} ({})", pool.pair_id, pool.pool_address);

        // Store in database
        let key = format!("pool:{}", pool.pool_address);
        let value = bincode::serialize(&pool)?;
        self.db.put(key.as_bytes(), value)?;

        // Update cache
        self.pool_cache.write().await.insert(pool.pool_address.clone(), pool.clone());

        // Update token's pool associations
        self.associate_pool_with_tokens(&pool).await?;

        info!("✅ Pool registered: {}", pool.pair_id);
        Ok(())
    }

    /// Associate a pool with its tokens
    async fn associate_pool_with_tokens(&self, pool: &PoolMetadata) -> Result<()> {
        // Update base token
        self.update_token_metadata(&pool.base_token_address, |token| {
            token.has_liquidity_pool = true;
            if !token.liquidity_pools.contains(&pool.pool_id) {
                token.liquidity_pools.push(pool.pool_id.clone());
            }
        }).await?;

        // Update quote token
        self.update_token_metadata(&pool.quote_token_address, |token| {
            token.has_liquidity_pool = true;
            if !token.liquidity_pools.contains(&pool.pool_id) {
                token.liquidity_pools.push(pool.pool_id.clone());
            }
        }).await?;

        // Update address-to-pools index
        let mut index = self.address_to_pools.write().await;
        index.entry(pool.base_token_address.clone())
            .or_insert_with(Vec::new)
            .push(pool.pool_address.clone());
        index.entry(pool.quote_token_address.clone())
            .or_insert_with(Vec::new)
            .push(pool.pool_address.clone());

        Ok(())
    }

    /// Get pool by address
    pub async fn get_pool_by_address(&self, address: &str) -> Result<Option<PoolMetadata>> {
        // Check cache first
        if let Some(pool) = self.pool_cache.read().await.get(address) {
            return Ok(Some(pool.clone()));
        }

        // Load from database
        let key = format!("pool:{}", address);
        if let Some(value) = self.db.get(key.as_bytes())? {
            let pool: PoolMetadata = bincode::deserialize(&value)?;

            // Update cache
            self.pool_cache.write().await.insert(address.to_string(), pool.clone());

            Ok(Some(pool))
        } else {
            Ok(None)
        }
    }

    /// Get pool by pair ID (e.g., "ORB/MYTOKEN")
    pub async fn get_pool_by_pair(&self, pair_id: &str) -> Result<Option<PoolMetadata>> {
        let cache = self.pool_cache.read().await;
        Ok(cache.values().find(|p| p.pair_id == pair_id).cloned())
    }

    /// Get all pools
    pub async fn get_all_pools(&self) -> Result<Vec<PoolMetadata>> {
        let cache = self.pool_cache.read().await;
        let mut pools: Vec<PoolMetadata> = cache.values().cloned().collect();

        // Sort by liquidity (highest first)
        pools.sort_by(|a, b| b.liquidity_usd.partial_cmp(&a.liquidity_usd).unwrap_or(std::cmp::Ordering::Equal));

        Ok(pools)
    }

    /// Get pools for a specific token
    pub async fn get_pools_for_token(&self, token_address: &str) -> Result<Vec<PoolMetadata>> {
        let pool_addresses = self.address_to_pools.read().await
            .get(token_address)
            .cloned()
            .unwrap_or_default();

        let mut pools = Vec::new();
        for address in pool_addresses {
            if let Some(pool) = self.get_pool_by_address(&address).await? {
                pools.push(pool);
            }
        }

        Ok(pools)
    }

    /// Update pool reserves after a trade
    pub async fn update_pool_reserves(&self, pool_address: &str, reserve_base: BigDecimal, reserve_quote: BigDecimal) -> Result<()> {
        if let Some(mut pool) = self.get_pool_by_address(pool_address).await? {
            pool.reserve_base = reserve_base;
            pool.reserve_quote = reserve_quote;
            pool.last_updated = Utc::now();
            self.register_pool(pool).await?;
        }
        Ok(())
    }

    // ============ TRADING PAIR OPERATIONS ============

    /// Register or update a trading pair
    pub async fn update_trading_pair(&self, pair: TradingPairMetadata) -> Result<()> {
        debug!("📊 Updating trading pair: {}", pair.pair_id);

        // Store in database
        let key = format!("pair:{}", pair.pair_id);
        let value = bincode::serialize(&pair)?;
        self.db.put(key.as_bytes(), value)?;

        // Update cache
        self.pair_cache.write().await.insert(pair.pair_id.clone(), pair);

        Ok(())
    }

    /// Get trading pair by ID
    pub async fn get_trading_pair(&self, pair_id: &str) -> Result<Option<TradingPairMetadata>> {
        // Check cache first
        if let Some(pair) = self.pair_cache.read().await.get(pair_id) {
            return Ok(Some(pair.clone()));
        }

        // Load from database
        let key = format!("pair:{}", pair_id);
        if let Some(value) = self.db.get(key.as_bytes())? {
            let pair: TradingPairMetadata = bincode::deserialize(&value)?;

            // Update cache
            self.pair_cache.write().await.insert(pair_id.to_string(), pair.clone());

            Ok(Some(pair))
        } else {
            Ok(None)
        }
    }

    /// Get all trading pairs
    pub async fn get_all_trading_pairs(&self) -> Result<Vec<TradingPairMetadata>> {
        let cache = self.pair_cache.read().await;
        let mut pairs: Vec<TradingPairMetadata> = cache.values().cloned().collect();

        // Sort by volume (highest first)
        pairs.sort_by(|a, b| b.volume_24h.partial_cmp(&a.volume_24h).unwrap_or(std::cmp::Ordering::Equal));

        Ok(pairs)
    }

    // ============ DATABASE LOADING OPERATIONS ============

    /// Load all tokens from database into cache
    async fn load_tokens_from_db(&self) -> Result<()> {
        let prefix = b"token:";
        let iter = self.db.iterator(IteratorMode::From(prefix, rocksdb::Direction::Forward));

        let mut count = 0;
        for item in iter {
            let (key, value) = item?;

            // Check if key starts with our prefix
            if !key.starts_with(prefix) {
                break;
            }

            if let Ok(token) = bincode::deserialize::<TokenMetadata>(&value) {
                let address = token.contract_address.clone();
                let symbol = token.symbol.clone();

                self.token_cache.write().await.insert(address.clone(), token);
                self.symbol_to_address.write().await.insert(symbol, address);
                count += 1;
            }
        }

        debug!("Loaded {} tokens from database", count);
        Ok(())
    }

    /// Load all pools from database into cache
    async fn load_pools_from_db(&self) -> Result<()> {
        let prefix = b"pool:";
        let iter = self.db.iterator(IteratorMode::From(prefix, rocksdb::Direction::Forward));

        let mut count = 0;
        for item in iter {
            let (key, value) = item?;

            if !key.starts_with(prefix) {
                break;
            }

            if let Ok(pool) = bincode::deserialize::<PoolMetadata>(&value) {
                let address = pool.pool_address.clone();
                self.pool_cache.write().await.insert(address, pool);
                count += 1;
            }
        }

        debug!("Loaded {} pools from database", count);
        Ok(())
    }

    /// Load all trading pairs from database into cache
    async fn load_pairs_from_db(&self) -> Result<()> {
        let prefix = b"pair:";
        let iter = self.db.iterator(IteratorMode::From(prefix, rocksdb::Direction::Forward));

        let mut count = 0;
        for item in iter {
            let (key, value) = item?;

            if !key.starts_with(prefix) {
                break;
            }

            if let Ok(pair) = bincode::deserialize::<TradingPairMetadata>(&value) {
                let pair_id = pair.pair_id.clone();
                self.pair_cache.write().await.insert(pair_id, pair);
                count += 1;
            }
        }

        debug!("Loaded {} trading pairs from database", count);
        Ok(())
    }

    /// Rebuild index mappings from cached data
    async fn rebuild_indices(&self) -> Result<()> {
        let mut address_to_pools = HashMap::new();

        // Build address-to-pools index
        for pool in self.pool_cache.read().await.values() {
            address_to_pools.entry(pool.base_token_address.clone())
                .or_insert_with(Vec::new)
                .push(pool.pool_address.clone());
            address_to_pools.entry(pool.quote_token_address.clone())
                .or_insert_with(Vec::new)
                .push(pool.pool_address.clone());
        }

        *self.address_to_pools.write().await = address_to_pools;

        Ok(())
    }
}

#[cfg(target_os = "windows")]
impl TokenRegistry {
    /// Create a new token registry (Windows stub - in-memory only)
    pub fn new(_db: Arc<()>) -> Self {
        Self::new_stub()
    }

    pub fn new_stub() -> Self {
        Self {
            token_cache: Arc::new(RwLock::new(HashMap::new())),
            pool_cache: Arc::new(RwLock::new(HashMap::new())),
            pair_cache: Arc::new(RwLock::new(HashMap::new())),
            symbol_to_address: Arc::new(RwLock::new(HashMap::new())),
            address_to_pools: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn initialize(&self) -> Result<()> { Ok(()) }
    pub async fn register_token(&self, _token: TokenMetadata) -> Result<()> { Ok(()) }
    pub async fn get_token_by_address(&self, _address: &str) -> Result<Option<TokenMetadata>> { Ok(None) }
    pub async fn get_token_by_symbol(&self, _symbol: &str) -> Result<Option<TokenMetadata>> { Ok(None) }
    pub async fn get_all_tokens(&self) -> Result<Vec<TokenMetadata>> { Ok(vec![]) }
    pub async fn get_active_tokens(&self) -> Result<Vec<TokenMetadata>> { Ok(vec![]) }
    pub async fn update_token_metadata(&self, _address: &str, _update_fn: impl FnOnce(&mut TokenMetadata)) -> Result<()> { Ok(()) }
    pub async fn update_token_price(&self, _address: &str, _price_usd: BigDecimal, _volume_24h: BigDecimal) -> Result<()> { Ok(()) }
    pub async fn register_pool(&self, _pool: PoolMetadata) -> Result<()> { Ok(()) }
    pub async fn get_pool_by_address(&self, _address: &str) -> Result<Option<PoolMetadata>> { Ok(None) }
    pub async fn get_pool_by_pair(&self, _pair_id: &str) -> Result<Option<PoolMetadata>> { Ok(None) }
    pub async fn get_all_pools(&self) -> Result<Vec<PoolMetadata>> { Ok(vec![]) }
    pub async fn get_pools_for_token(&self, _address: &str) -> Result<Vec<PoolMetadata>> { Ok(vec![]) }
    pub async fn update_pool_reserves(&self, _pool_address: &str, _reserve_base: BigDecimal, _reserve_quote: BigDecimal) -> Result<()> { Ok(()) }
    pub async fn update_trading_pair(&self, _pair: TradingPairMetadata) -> Result<()> { Ok(()) }
    pub async fn get_trading_pair(&self, _pair_id: &str) -> Result<Option<TradingPairMetadata>> { Ok(None) }
    pub async fn get_all_trading_pairs(&self) -> Result<Vec<TradingPairMetadata>> { Ok(vec![]) }
}

#[cfg(test)]
#[cfg(not(target_os = "windows"))]
mod tests {
    use super::*;
    use tempfile::tempdir;

    async fn create_test_registry() -> TokenRegistry {
        let temp_dir = tempdir().unwrap();
        let db = Arc::new(DB::open_default(temp_dir.path()).unwrap());
        let registry = TokenRegistry::new(db);
        registry.initialize().await.unwrap();
        registry
    }

    #[tokio::test]
    async fn test_token_registration() {
        let registry = create_test_registry().await;

        let token = TokenMetadata {
            contract_address: "0x123".to_string(),
            symbol: "TEST".to_string(),
            name: "Test Token".to_string(),
            decimals: 18,
            total_supply: BigDecimal::from(1000000),
            circulating_supply: BigDecimal::from(1000000),
            creator: "creator1".to_string(),
            created_at: Utc::now(),
            is_verified: false,
            is_active: true,
            price_usd: BigDecimal::from(0),
            market_cap: BigDecimal::from(0),
            volume_24h: BigDecimal::from(0),
            price_change_24h_bps: 0,
            logo_url: None,
            website: None,
            description: None,
            tags: vec![],
            has_liquidity_pool: false,
            liquidity_pools: vec![],
            last_updated: Utc::now(),
            social_links: None, // v2.7.7-beta
        };

        registry.register_token(token.clone()).await.unwrap();

        let retrieved = registry.get_token_by_address("0x123").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().symbol, "TEST");

        let by_symbol = registry.get_token_by_symbol("TEST").await.unwrap();
        assert!(by_symbol.is_some());
        assert_eq!(by_symbol.unwrap().contract_address, "0x123");
    }

    #[tokio::test]
    async fn test_pool_registration() {
        let registry = create_test_registry().await;

        let pool = PoolMetadata {
            pool_id: "test-pool".to_string(),
            pool_address: "0xpool".to_string(),
            pair_id: "TEST/ORB".to_string(),
            base_token: "TEST".to_string(),
            quote_token: "ORB".to_string(),
            base_token_address: "0x123".to_string(),
            quote_token_address: "0x456".to_string(),
            reserve_base: BigDecimal::from(1000),
            reserve_quote: BigDecimal::from(1000),
            total_shares: BigDecimal::from(1000),
            fee_rate: BigDecimal::from(0.003),
            created_at: Utc::now(),
            creator: "creator1".to_string(),
            is_active: true,
            is_paused: false,
            liquidity_locked_until: None,
            volume_24h: BigDecimal::from(0),
            fees_24h: BigDecimal::from(0),
            liquidity_usd: BigDecimal::from(0),
            apr: 0.0,
            provider_count: 0,
            last_updated: Utc::now(),
        };

        registry.register_pool(pool.clone()).await.unwrap();

        let retrieved = registry.get_pool_by_address("0xpool").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().pair_id, "TEST/ORB");
    }
}
