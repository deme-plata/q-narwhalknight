/// Distributed DEX Protocol for Q-NarwhalKnight
/// Enables decentralized order book management and trade matching across all nodes
///
/// Features:
/// - Order book synchronization via libp2p gossipsub
/// - Distributed trade matching
/// - Liquidity pool state sharing
/// - Cross-node arbitrage detection

use anyhow::{anyhow, Result};
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Topics for DEX gossip
pub const TOPIC_ORDER_BOOK: &str = "qnk/dex/order-book/v1";
pub const TOPIC_TRADE_EXECUTION: &str = "qnk/dex/trade/v1";
pub const TOPIC_LIQUIDITY_POOL: &str = "qnk/dex/liquidity/v1";
pub const TOPIC_PRICE_UPDATE: &str = "qnk/dex/price/v1";

/// Order book update message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookMessage {
    pub trading_pair: TradingPair,
    pub orders: Vec<Order>,
    pub timestamp: u64,
    pub node_id: String,
}

/// Trading pair identifier
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TradingPair {
    pub base: String,  // e.g., "QNK"
    pub quote: String, // e.g., "USDT"
}

/// Order in the order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub order_id: String,
    pub trader: [u8; 32],
    pub order_type: OrderType,
    pub side: OrderSide,
    pub price: u64,        // In base units
    pub amount: u64,       // In base units
    pub filled: u64,       // Amount filled
    pub timestamp: u64,
    pub expires_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderType {
    Limit,
    Market,
    StopLimit { stop_price: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Trade execution message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeMessage {
    pub trade_id: String,
    pub trading_pair: TradingPair,
    pub buy_order_id: String,
    pub sell_order_id: String,
    pub price: u64,
    pub amount: u128,
    pub buyer: [u8; 32],
    pub seller: [u8; 32],
    pub timestamp: u64,
    pub executor_node: String,
}

/// Liquidity pool state message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityPoolMessage {
    pub pool_address: [u8; 32],
    pub token_a: String,
    pub token_b: String,
    pub reserve_a: u128,
    pub reserve_b: u128,
    pub total_liquidity: u128,
    pub fee_rate: u32, // In basis points (100 = 1%)
    pub last_update: u64,
}

/// Price update message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceUpdateMessage {
    pub trading_pair: TradingPair,
    pub price: u64,
    pub volume_24h: u64,
    pub change_24h: i64, // Basis points
    pub timestamp: u64,
}

/// Distributed DEX coordinator
pub struct DistributedDEXCoordinator {
    /// Local peer ID
    pub local_peer_id: PeerId,
    /// Order books indexed by trading pair
    pub order_books: Arc<RwLock<HashMap<TradingPair, OrderBook>>>,
    /// Liquidity pools indexed by pool address
    pub liquidity_pools: Arc<RwLock<HashMap<[u8; 32], LiquidityPool>>>,
    /// Recent trades for analytics
    pub recent_trades: Arc<RwLock<Vec<TradeMessage>>>,
    /// Price cache
    pub prices: Arc<RwLock<HashMap<TradingPair, u64>>>,
    /// Trading statistics
    pub stats: Arc<RwLock<DEXStats>>,
}

/// Order book for a trading pair
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub trading_pair: TradingPair,
    pub bids: BTreeMap<u64, Vec<Order>>, // Price -> Orders (descending)
    pub asks: BTreeMap<u64, Vec<Order>>, // Price -> Orders (ascending)
    pub last_updated: u64,
}

/// Liquidity pool state
#[derive(Debug, Clone)]
pub struct LiquidityPool {
    pub address: [u8; 32],
    pub token_a: String,
    pub token_b: String,
    pub reserve_a: u128,
    pub reserve_b: u128,
    pub total_liquidity: u128,
    pub fee_rate: u32,
}

/// DEX statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DEXStats {
    pub total_orders: u64,
    pub total_trades: u64,
    pub total_volume_24h: u128,
    pub active_pairs: u64,
    pub active_pools: u64,
}

impl DistributedDEXCoordinator {
    pub fn new(local_peer_id: PeerId) -> Self {
        info!("💱 Initializing Distributed DEX Coordinator");
        Self {
            local_peer_id,
            order_books: Arc::new(RwLock::new(HashMap::new())),
            liquidity_pools: Arc::new(RwLock::new(HashMap::new())),
            recent_trades: Arc::new(RwLock::new(Vec::new())),
            prices: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(DEXStats {
                total_orders: 0,
                total_trades: 0,
                total_volume_24h: 0,
                active_pairs: 0,
                active_pools: 0,
            })),
        }
    }

    /// Handle incoming order book update
    pub async fn handle_order_book_update(&self, msg: OrderBookMessage) -> Result<()> {
        debug!(
            "📋 Received order book update for {}/{}",
            msg.trading_pair.base, msg.trading_pair.quote
        );

        let mut books = self.order_books.write().await;

        let book = books
            .entry(msg.trading_pair.clone())
            .or_insert_with(|| OrderBook {
                trading_pair: msg.trading_pair.clone(),
                bids: BTreeMap::new(),
                asks: BTreeMap::new(),
                last_updated: 0,
            });

        // Only update if this is newer
        if msg.timestamp > book.last_updated {
            // Clear old orders
            book.bids.clear();
            book.asks.clear();

            // Insert new orders
            for order in msg.orders {
                match order.side {
                    OrderSide::Buy => {
                        book.bids
                            .entry(order.price)
                            .or_insert_with(Vec::new)
                            .push(order);
                    }
                    OrderSide::Sell => {
                        book.asks
                            .entry(order.price)
                            .or_insert_with(Vec::new)
                            .push(order);
                    }
                }
            }

            book.last_updated = msg.timestamp;

            // Update stats
            let mut stats = self.stats.write().await;
            stats.active_pairs = books.len() as u64;

            info!("✅ Updated order book for {}/{}", msg.trading_pair.base, msg.trading_pair.quote);
        }

        Ok(())
    }

    /// Handle trade execution message
    pub async fn handle_trade(&self, msg: TradeMessage) -> Result<()> {
        debug!(
            "💸 Trade executed: {} {} at price {}",
            msg.amount, msg.trading_pair.base, msg.price
        );

        // Update recent trades
        let mut trades = self.recent_trades.write().await;
        trades.push(msg.clone());

        // Keep only last 1000 trades
        if trades.len() > 1000 {
            trades.remove(0);
        }

        // Update price
        let mut prices = self.prices.write().await;
        prices.insert(msg.trading_pair.clone(), msg.price);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_trades += 1;
        stats.total_volume_24h += msg.amount;

        Ok(())
    }

    /// Handle liquidity pool update
    pub async fn handle_liquidity_pool(&self, msg: LiquidityPoolMessage) -> Result<()> {
        debug!(
            "🏊 Liquidity pool update: {}/{} reserves: {}/{}",
            msg.token_a, msg.token_b, msg.reserve_a, msg.reserve_b
        );

        let mut pools = self.liquidity_pools.write().await;

        pools.insert(
            msg.pool_address,
            LiquidityPool {
                address: msg.pool_address,
                token_a: msg.token_a,
                token_b: msg.token_b,
                reserve_a: msg.reserve_a,
                reserve_b: msg.reserve_b,
                total_liquidity: msg.total_liquidity,
                fee_rate: msg.fee_rate,
            },
        );

        // Update stats
        let mut stats = self.stats.write().await;
        stats.active_pools = pools.len() as u64;

        Ok(())
    }

    /// Get order book for a trading pair
    pub async fn get_order_book(&self, pair: &TradingPair) -> Option<OrderBook> {
        let books = self.order_books.read().await;
        books.get(pair).cloned()
    }

    /// Get best bid/ask prices
    pub async fn get_best_prices(&self, pair: &TradingPair) -> Option<(u64, u64)> {
        let books = self.order_books.read().await;

        if let Some(book) = books.get(pair) {
            let best_bid = book.bids.keys().next_back().copied();
            let best_ask = book.asks.keys().next().copied();

            if let (Some(bid), Some(ask)) = (best_bid, best_ask) {
                return Some((bid, ask));
            }
        }

        None
    }

    /// Calculate liquidity pool price
    pub async fn get_pool_price(&self, pool_address: &[u8; 32]) -> Option<u64> {
        let pools = self.liquidity_pools.read().await;

        if let Some(pool) = pools.get(pool_address) {
            if pool.reserve_b > 0 {
                // Price = reserve_a / reserve_b (in base units)
                return Some((pool.reserve_a as u128 * 1_000_000_000 / pool.reserve_b as u128) as u64);
            }
        }

        None
    }

    /// Detect arbitrage opportunities between order books and pools
    pub async fn detect_arbitrage(&self) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();

        let books = self.order_books.read().await;
        let pools = self.liquidity_pools.read().await;

        // Compare order book prices with pool prices
        for (pair, book) in books.iter() {
            if let (Some(best_bid), Some(best_ask)) = (
                book.bids.keys().next_back(),
                book.asks.keys().next(),
            ) {
                // Look for matching pools
                for (pool_address, pool) in pools.iter() {
                    // Check if pool matches the trading pair
                    if (pool.token_a == pair.base && pool.token_b == pair.quote)
                        || (pool.token_a == pair.quote && pool.token_b == pair.base)
                    {
                        if let Some(pool_price) = self.get_pool_price(pool_address).await {
                            // Check for arbitrage: buy from pool, sell on order book
                            if pool_price < *best_bid {
                                let profit_bps = ((*best_bid - pool_price) as f64 / pool_price as f64 * 10000.0) as i64;

                                if profit_bps > 100 {
                                    // More than 1% profit
                                    opportunities.push(ArbitrageOpportunity {
                                        pair: pair.clone(),
                                        pool_address: *pool_address,
                                        pool_price,
                                        order_book_price: *best_bid,
                                        profit_bps,
                                        direction: ArbitrageDirection::PoolToBook,
                                    });
                                }
                            }

                            // Check reverse: buy from order book, sell to pool
                            if *best_ask < pool_price {
                                let profit_bps = ((pool_price - *best_ask) as f64 / *best_ask as f64 * 10000.0) as i64;

                                if profit_bps > 100 {
                                    opportunities.push(ArbitrageOpportunity {
                                        pair: pair.clone(),
                                        pool_address: *pool_address,
                                        pool_price,
                                        order_book_price: *best_ask,
                                        profit_bps,
                                        direction: ArbitrageDirection::BookToPool,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        if !opportunities.is_empty() {
            info!("🎯 Detected {} arbitrage opportunities", opportunities.len());
        }

        opportunities
    }

    /// Get DEX statistics
    pub async fn get_stats(&self) -> DEXStats {
        self.stats.read().await.clone()
    }

    /// Broadcast order book update
    pub async fn broadcast_order_book(&self, pair: TradingPair, orders: Vec<Order>) -> Result<OrderBookMessage> {
        let msg = OrderBookMessage {
            trading_pair: pair,
            orders,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            node_id: self.local_peer_id.to_string(),
        };

        info!("📡 Broadcasting order book update");

        Ok(msg)
    }
}

/// Arbitrage opportunity
#[derive(Debug, Clone)]
pub struct ArbitrageOpportunity {
    pub pair: TradingPair,
    pub pool_address: [u8; 32],
    pub pool_price: u64,
    pub order_book_price: u64,
    pub profit_bps: i64, // Basis points (100 = 1%)
    pub direction: ArbitrageDirection,
}

#[derive(Debug, Clone)]
pub enum ArbitrageDirection {
    PoolToBook, // Buy from pool, sell on order book
    BookToPool, // Buy from order book, sell to pool
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dex_coordinator_creation() {
        let peer_id = PeerId::random();
        let coordinator = DistributedDEXCoordinator::new(peer_id);

        let stats = coordinator.get_stats().await;
        assert_eq!(stats.total_trades, 0);
        assert_eq!(stats.active_pairs, 0);
    }

    #[tokio::test]
    async fn test_order_book_update() {
        let peer_id = PeerId::random();
        let coordinator = DistributedDEXCoordinator::new(peer_id);

        let pair = TradingPair {
            base: "QNK".to_string(),
            quote: "USDT".to_string(),
        };

        let msg = OrderBookMessage {
            trading_pair: pair.clone(),
            orders: vec![],
            timestamp: 1000,
            node_id: peer_id.to_string(),
        };

        coordinator.handle_order_book_update(msg).await.unwrap();

        let book = coordinator.get_order_book(&pair).await;
        assert!(book.is_some());
    }
}
