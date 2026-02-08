//! Price History Tracking System
//!
//! Time-series database for tracking token prices, trades, and OHLCV data.
//! Provides historical price data for charting and analytics.

use anyhow::Result;
use bigdecimal::BigDecimal;
use chrono::{DateTime, Duration, Utc};
#[cfg(not(target_os = "windows"))]
use rocksdb::{IteratorMode, DB};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Candle interval for OHLCV data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CandleInterval {
    Minute1,
    Minute5,
    Minute15,
    Hour1,
    Hour4,
    Day1,
    Week1,
}

impl CandleInterval {
    /// Get duration in seconds
    pub fn duration_seconds(&self) -> i64 {
        match self {
            CandleInterval::Minute1 => 60,
            CandleInterval::Minute5 => 300,
            CandleInterval::Minute15 => 900,
            CandleInterval::Hour1 => 3600,
            CandleInterval::Hour4 => 14400,
            CandleInterval::Day1 => 86400,
            CandleInterval::Week1 => 604800,
        }
    }

    /// Get string representation
    pub fn as_str(&self) -> &str {
        match self {
            CandleInterval::Minute1 => "1m",
            CandleInterval::Minute5 => "5m",
            CandleInterval::Minute15 => "15m",
            CandleInterval::Hour1 => "1h",
            CandleInterval::Hour4 => "4h",
            CandleInterval::Day1 => "1d",
            CandleInterval::Week1 => "1w",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "1m" => Some(CandleInterval::Minute1),
            "5m" => Some(CandleInterval::Minute5),
            "15m" => Some(CandleInterval::Minute15),
            "1h" => Some(CandleInterval::Hour1),
            "4h" => Some(CandleInterval::Hour4),
            "1d" => Some(CandleInterval::Day1),
            "1w" => Some(CandleInterval::Week1),
            _ => None,
        }
    }
}

/// OHLCV candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVCandle {
    pub pair_id: String,
    pub interval: CandleInterval,
    pub timestamp: DateTime<Utc>, // Start of candle period
    pub open: BigDecimal,
    pub high: BigDecimal,
    pub low: BigDecimal,
    pub close: BigDecimal,
    pub volume: BigDecimal,
    pub trades_count: u64,
}

/// Individual trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub trade_id: String,
    pub pair_id: String,
    pub timestamp: DateTime<Utc>,
    pub price: BigDecimal,
    pub amount: BigDecimal,
    pub side: TradeSide, // Buy or Sell
    pub trader: Option<String>, // Optional for privacy
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Price history manager
#[cfg(not(target_os = "windows"))]
pub struct PriceHistoryManager {
    db: Arc<DB>,

    // In-memory cache for recent candles (last 24 hours)
    recent_candles: Arc<RwLock<HashMap<String, HashMap<CandleInterval, Vec<OHLCVCandle>>>>>,

    // Active candles being built (not yet closed)
    active_candles: Arc<RwLock<HashMap<String, HashMap<CandleInterval, OHLCVCandle>>>>,
}

#[cfg(target_os = "windows")]
pub struct PriceHistoryManager {
    // In-memory cache for recent candles (last 24 hours)
    recent_candles: Arc<RwLock<HashMap<String, HashMap<CandleInterval, Vec<OHLCVCandle>>>>>,

    // Active candles being built (not yet closed)
    active_candles: Arc<RwLock<HashMap<String, HashMap<CandleInterval, OHLCVCandle>>>>,
}

#[cfg(not(target_os = "windows"))]
impl PriceHistoryManager {
    /// Create a new price history manager
    pub fn new(db: Arc<DB>) -> Self {
        Self {
            db,
            recent_candles: Arc::new(RwLock::new(HashMap::new())),
            active_candles: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Initialize the price history manager
    pub async fn initialize(&self) -> Result<()> {
        info!("📈 Initializing Price History Manager");

        // Load recent candles (last 24 hours) for all intervals
        self.load_recent_candles().await?;

        info!("✅ Price History Manager initialized");
        Ok(())
    }

    /// Record a trade and update all relevant candles
    pub async fn record_trade(&self, trade: TradeRecord) -> Result<()> {
        debug!("📝 Recording trade: {} @ {} for {}", trade.pair_id, trade.price, trade.amount);

        // Store trade record
        self.store_trade_record(&trade).await?;

        // Update candles for all intervals
        for interval in [
            CandleInterval::Minute1,
            CandleInterval::Minute5,
            CandleInterval::Minute15,
            CandleInterval::Hour1,
            CandleInterval::Hour4,
            CandleInterval::Day1,
            CandleInterval::Week1,
        ] {
            self.update_candle_with_trade(&trade, interval).await?;
        }

        Ok(())
    }

    /// Update a candle with a new trade
    async fn update_candle_with_trade(&self, trade: &TradeRecord, interval: CandleInterval) -> Result<()> {
        let candle_timestamp = self.get_candle_timestamp(&trade.timestamp, interval);
        let cache_key = self.get_candle_key(&trade.pair_id, interval, &candle_timestamp);

        let mut active_candles = self.active_candles.write().await;
        let pair_candles = active_candles.entry(trade.pair_id.clone()).or_insert_with(HashMap::new);

        let candle = pair_candles.entry(interval).or_insert_with(|| OHLCVCandle {
            pair_id: trade.pair_id.clone(),
            interval,
            timestamp: candle_timestamp,
            open: trade.price.clone(),
            high: trade.price.clone(),
            low: trade.price.clone(),
            close: trade.price.clone(),
            volume: BigDecimal::from(0),
            trades_count: 0,
        });

        // Update OHLCV data
        if &trade.price > &candle.high {
            candle.high = trade.price.clone();
        }
        if &trade.price < &candle.low {
            candle.low = trade.price.clone();
        }
        candle.close = trade.price.clone();
        candle.volume += &trade.amount;
        candle.trades_count += 1;

        // Check if candle should be closed
        let now = Utc::now();
        let candle_end = candle.timestamp + Duration::seconds(interval.duration_seconds());

        if now >= candle_end {
            // Close and store candle
            let closed_candle = candle.clone();
            self.store_candle(&closed_candle).await?;

            // Remove from active candles
            pair_candles.remove(&interval);

            // Add to recent candles cache
            let mut recent = self.recent_candles.write().await;
            recent.entry(trade.pair_id.clone())
                .or_insert_with(HashMap::new)
                .entry(interval)
                .or_insert_with(Vec::new)
                .push(closed_candle);
        }

        Ok(())
    }

    /// Get candle timestamp (start of period)
    fn get_candle_timestamp(&self, timestamp: &DateTime<Utc>, interval: CandleInterval) -> DateTime<Utc> {
        let duration_secs = interval.duration_seconds();
        let ts_secs = timestamp.timestamp();
        let candle_start = (ts_secs / duration_secs) * duration_secs;
        DateTime::from_timestamp(candle_start, 0).unwrap_or(*timestamp)
    }

    /// Get unique key for candle storage
    fn get_candle_key(&self, pair_id: &str, interval: CandleInterval, timestamp: &DateTime<Utc>) -> String {
        format!("candle:{}:{}:{}", pair_id, interval.as_str(), timestamp.timestamp())
    }

    /// Store a completed candle in the database
    async fn store_candle(&self, candle: &OHLCVCandle) -> Result<()> {
        let key = self.get_candle_key(&candle.pair_id, candle.interval, &candle.timestamp);
        let value = bincode::serialize(candle)?;
        self.db.put(key.as_bytes(), value)?;
        Ok(())
    }

    /// Store a trade record
    async fn store_trade_record(&self, trade: &TradeRecord) -> Result<()> {
        let key = format!("trade:{}:{}", trade.pair_id, trade.timestamp.timestamp_millis());
        let value = bincode::serialize(trade)?;
        self.db.put(key.as_bytes(), value)?;
        Ok(())
    }

    /// Get historical candles for a pair and interval
    pub async fn get_historical_candles(
        &self,
        pair_id: &str,
        interval: CandleInterval,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
        limit: Option<usize>,
    ) -> Result<Vec<OHLCVCandle>> {
        let mut candles = Vec::new();

        // Generate all candle timestamps in range
        let duration_secs = interval.duration_seconds();
        let mut current = self.get_candle_timestamp(&from, interval);

        while current <= to {
            let key = self.get_candle_key(pair_id, interval, &current);

            if let Some(value) = self.db.get(key.as_bytes())? {
                if let Ok(candle) = bincode::deserialize::<OHLCVCandle>(&value) {
                    candles.push(candle);

                    if let Some(limit_val) = limit {
                        if candles.len() >= limit_val {
                            break;
                        }
                    }
                }
            }

            current = current + Duration::seconds(duration_secs);
        }

        Ok(candles)
    }

    /// Get recent candles from cache (faster than DB query)
    pub async fn get_recent_candles(
        &self,
        pair_id: &str,
        interval: CandleInterval,
        limit: usize,
    ) -> Result<Vec<OHLCVCandle>> {
        let cache = self.recent_candles.read().await;

        if let Some(pair_data) = cache.get(pair_id) {
            if let Some(interval_candles) = pair_data.get(&interval) {
                let start = if interval_candles.len() > limit {
                    interval_candles.len() - limit
                } else {
                    0
                };
                return Ok(interval_candles[start..].to_vec());
            }
        }

        // Fall back to database query
        let to = Utc::now();
        let from = to - Duration::hours(24); // Last 24 hours
        self.get_historical_candles(pair_id, interval, from, to, Some(limit)).await
    }

    /// Get latest price for a trading pair
    pub async fn get_latest_price(&self, pair_id: &str) -> Result<Option<BigDecimal>> {
        // Check active candles first
        let active = self.active_candles.read().await;
        if let Some(pair_candles) = active.get(pair_id) {
            if let Some(candle) = pair_candles.get(&CandleInterval::Minute1) {
                return Ok(Some(candle.close.clone()));
            }
        }

        // Check recent candles
        let recent = self.recent_candles.read().await;
        if let Some(pair_data) = recent.get(pair_id) {
            if let Some(candles) = pair_data.get(&CandleInterval::Minute1) {
                if let Some(last_candle) = candles.last() {
                    return Ok(Some(last_candle.close.clone()));
                }
            }
        }

        Ok(None)
    }

    /// Get 24-hour statistics for a trading pair
    pub async fn get_24h_stats(&self, pair_id: &str) -> Result<Option<DayStatistics>> {
        let to = Utc::now();
        let from = to - Duration::hours(24);

        let candles = self.get_historical_candles(
            pair_id,
            CandleInterval::Minute5, // Use 5-minute candles for 24h stats
            from,
            to,
            None,
        ).await?;

        if candles.is_empty() {
            return Ok(None);
        }

        let mut high = candles[0].high.clone();
        let mut low = candles[0].low.clone();
        let mut volume = BigDecimal::from(0);
        let mut trades_count = 0;

        for candle in &candles {
            if candle.high > high {
                high = candle.high.clone();
            }
            if candle.low < low {
                low = candle.low.clone();
            }
            volume += &candle.volume;
            trades_count += candle.trades_count;
        }

        let open = candles.first().unwrap().open.clone();
        let close = candles.last().unwrap().close.clone();

        // Calculate price change percentage
        let price_change = if open > BigDecimal::from(0) {
            let change = &close - &open;
            (change / &open).to_string().parse().unwrap_or(0.0) * 100.0
        } else {
            0.0
        };

        Ok(Some(DayStatistics {
            pair_id: pair_id.to_string(),
            open,
            close,
            high,
            low,
            volume,
            trades_count,
            price_change_percent: price_change,
            timestamp: to,
        }))
    }

    /// Load recent candles from database (last 24 hours)
    async fn load_recent_candles(&self) -> Result<()> {
        let to = Utc::now();
        let from = to - Duration::hours(24);

        // For now, we'll load on-demand rather than preloading all pairs
        // This can be optimized by maintaining a list of active pairs

        Ok(())
    }

    /// Cleanup old candles (retention policy)
    pub async fn cleanup_old_candles(&self, retention_days: i64) -> Result<u64> {
        let cutoff = Utc::now() - Duration::days(retention_days);
        let cutoff_ts = cutoff.timestamp();

        let prefix = b"candle:";
        let iter = self.db.iterator(IteratorMode::From(prefix, rocksdb::Direction::Forward));

        let mut deleted_count = 0;

        for item in iter {
            let (key, _value) = item?;

            if !key.starts_with(prefix) {
                break;
            }

            // Extract timestamp from key (format: "candle:pair:interval:timestamp")
            if let Ok(key_str) = std::str::from_utf8(&key) {
                let parts: Vec<&str> = key_str.split(':').collect();
                if parts.len() >= 4 {
                    if let Ok(ts) = parts[3].parse::<i64>() {
                        if ts < cutoff_ts {
                            self.db.delete(&key)?;
                            deleted_count += 1;
                        }
                    }
                }
            }
        }

        info!("🗑️  Cleaned up {} old candles (older than {} days)", deleted_count, retention_days);
        Ok(deleted_count)
    }
}

#[cfg(target_os = "windows")]
impl PriceHistoryManager {
    /// Create a new price history manager (Windows stub - in-memory only)
    pub fn new(_db: Arc<()>) -> Self {
        Self::new_stub()
    }

    pub fn new_stub() -> Self {
        Self {
            recent_candles: Arc::new(RwLock::new(HashMap::new())),
            active_candles: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn initialize(&self) -> Result<()> { Ok(()) }

    pub async fn record_trade(&self, _trade: TradeRecord) -> Result<()> { Ok(()) }

    pub fn get_candle_timestamp(&self, timestamp: &DateTime<Utc>, interval: CandleInterval) -> DateTime<Utc> {
        let secs = timestamp.timestamp();
        let interval_secs = interval.duration_seconds();
        let aligned = (secs / interval_secs) * interval_secs;
        DateTime::from_timestamp(aligned, 0).unwrap_or(*timestamp)
    }

    pub async fn get_historical_candles(&self, _pair_id: &str, _interval: CandleInterval, _start: DateTime<Utc>, _end: DateTime<Utc>, _limit: Option<usize>) -> Result<Vec<OHLCVCandle>> {
        Ok(vec![])
    }

    pub async fn get_recent_candles(&self, _pair_id: &str, _interval: CandleInterval, _count: usize) -> Result<Vec<OHLCVCandle>> {
        Ok(vec![])
    }

    pub async fn get_latest_price(&self, _pair_id: &str) -> Result<Option<BigDecimal>> { Ok(None) }

    pub async fn get_24h_stats(&self, _pair_id: &str) -> Result<Option<DayStatistics>> { Ok(None) }

    pub async fn cleanup_old_candles(&self, _retention_days: i64) -> Result<u64> { Ok(0) }
}

/// 24-hour trading statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DayStatistics {
    pub pair_id: String,
    pub open: BigDecimal,
    pub close: BigDecimal,
    pub high: BigDecimal,
    pub low: BigDecimal,
    pub volume: BigDecimal,
    pub trades_count: u64,
    pub price_change_percent: f64,
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
#[cfg(not(target_os = "windows"))]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_candle_timestamp_calculation() {
        let temp_dir = tempdir().unwrap();
        let db = Arc::new(DB::open_default(temp_dir.path()).unwrap());
        let manager = PriceHistoryManager::new(db);

        let timestamp = DateTime::from_timestamp(1700000000, 0).unwrap(); // Some timestamp

        let candle_ts_1m = manager.get_candle_timestamp(&timestamp, CandleInterval::Minute1);
        let candle_ts_1h = manager.get_candle_timestamp(&timestamp, CandleInterval::Hour1);

        // Candle timestamps should be aligned to interval boundaries
        assert_eq!(candle_ts_1m.timestamp() % 60, 0);
        assert_eq!(candle_ts_1h.timestamp() % 3600, 0);
    }

    #[tokio::test]
    async fn test_trade_recording() {
        let temp_dir = tempdir().unwrap();
        let db = Arc::new(DB::open_default(temp_dir.path()).unwrap());
        let manager = PriceHistoryManager::new(db);
        manager.initialize().await.unwrap();

        let trade = TradeRecord {
            trade_id: "trade1".to_string(),
            pair_id: "ORB/USD".to_string(),
            timestamp: Utc::now(),
            price: BigDecimal::from(100),
            amount: BigDecimal::from(10),
            side: TradeSide::Buy,
            trader: None,
        };

        manager.record_trade(trade).await.unwrap();

        // Check that active candles were updated
        let active = manager.active_candles.read().await;
        assert!(active.contains_key("ORB/USD"));
    }
}
