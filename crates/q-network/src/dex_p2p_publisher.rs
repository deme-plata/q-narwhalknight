/// DEX P2P Publisher - v2.9.2-beta
/// Publishes DEX events (trades, liquidity, prices) to gossipsub topics
/// for TRUE decentralization across all nodes
///
/// This module bridges the gap between local DEX operations and network-wide
/// synchronization, ensuring all nodes have consistent DEX state.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::distributed_dex::{
    LiquidityPoolMessage, OrderBookMessage, PriceUpdateMessage, TradeMessage, TradingPair,
    TOPIC_LIQUIDITY_POOL, TOPIC_ORDER_BOOK, TOPIC_PRICE_UPDATE, TOPIC_TRADE_EXECUTION,
};

/// Network command for publishing DEX events
#[derive(Debug, Clone)]
pub enum DexNetworkCommand {
    /// Publish a trade execution to all peers
    PublishTrade(TradeMessage),
    /// Publish liquidity pool update to all peers
    PublishLiquidityUpdate(LiquidityPoolMessage),
    /// Publish price update to all peers
    PublishPriceUpdate(PriceUpdateMessage),
    /// Publish order book update to all peers
    PublishOrderBook(OrderBookMessage),
}

/// Serialized DEX message for network transport
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexNetworkMessage {
    pub message_type: DexMessageType,
    pub payload: Vec<u8>,
    pub timestamp: u64,
    pub node_id: String,
    /// Signature for authenticity (Ed25519 or Dilithium)
    pub signature: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DexMessageType {
    Trade,
    LiquidityPool,
    PriceUpdate,
    OrderBook,
}

/// DEX P2P Publisher for broadcasting DEX events across the network
pub struct DexP2PPublisher {
    /// Channel sender for network commands
    network_cmd_tx: Option<mpsc::UnboundedSender<super::NetworkCommand>>,
    /// Local node ID for message attribution
    local_node_id: String,
    /// Statistics
    stats: DexPublisherStats,
}

#[derive(Debug, Default, Clone)]
pub struct DexPublisherStats {
    pub trades_published: u64,
    pub liquidity_updates_published: u64,
    pub price_updates_published: u64,
    pub order_book_updates_published: u64,
    pub publish_failures: u64,
}

impl DexP2PPublisher {
    /// Create a new DEX P2P Publisher
    pub fn new(local_node_id: String) -> Self {
        info!("💱 [DEX P2P] Initializing DEX P2P Publisher for node: {}", &local_node_id[..16.min(local_node_id.len())]);
        Self {
            network_cmd_tx: None,
            local_node_id,
            stats: DexPublisherStats::default(),
        }
    }

    /// Set the network command sender (called during initialization)
    pub fn set_network_sender(&mut self, tx: mpsc::UnboundedSender<super::NetworkCommand>) {
        self.network_cmd_tx = Some(tx);
        info!("💱 [DEX P2P] Network sender connected - DEX events will now be broadcast");
    }

    /// Publish a trade execution to all peers
    pub async fn publish_trade(&mut self, trade: TradeMessage) -> Result<()> {
        info!(
            "💱 [DEX P2P] Broadcasting trade: {} {} @ {} (pair: {}/{})",
            trade.amount,
            trade.trading_pair.base,
            trade.price,
            trade.trading_pair.base,
            trade.trading_pair.quote
        );

        let payload = postcard::to_allocvec(&trade)?;
        self.publish_to_topic(TOPIC_TRADE_EXECUTION, DexMessageType::Trade, payload).await?;
        self.stats.trades_published += 1;

        Ok(())
    }

    /// Publish liquidity pool update to all peers
    pub async fn publish_liquidity_update(&mut self, pool: LiquidityPoolMessage) -> Result<()> {
        info!(
            "🏊 [DEX P2P] Broadcasting liquidity update: {}/{} reserves: {}/{}",
            pool.token_a, pool.token_b, pool.reserve_a, pool.reserve_b
        );

        let payload = postcard::to_allocvec(&pool)?;
        self.publish_to_topic(TOPIC_LIQUIDITY_POOL, DexMessageType::LiquidityPool, payload).await?;
        self.stats.liquidity_updates_published += 1;

        Ok(())
    }

    /// Publish price update to all peers
    pub async fn publish_price_update(&mut self, price: PriceUpdateMessage) -> Result<()> {
        debug!(
            "📈 [DEX P2P] Broadcasting price update: {}/{} = {} (vol: {})",
            price.trading_pair.base, price.trading_pair.quote, price.price, price.volume_24h
        );

        let payload = postcard::to_allocvec(&price)?;
        self.publish_to_topic(TOPIC_PRICE_UPDATE, DexMessageType::PriceUpdate, payload).await?;
        self.stats.price_updates_published += 1;

        Ok(())
    }

    /// Publish order book update to all peers
    pub async fn publish_order_book(&mut self, order_book: OrderBookMessage) -> Result<()> {
        debug!(
            "📋 [DEX P2P] Broadcasting order book: {}/{} ({} orders)",
            order_book.trading_pair.base,
            order_book.trading_pair.quote,
            order_book.orders.len()
        );

        let payload = postcard::to_allocvec(&order_book)?;
        self.publish_to_topic(TOPIC_ORDER_BOOK, DexMessageType::OrderBook, payload).await?;
        self.stats.order_book_updates_published += 1;

        Ok(())
    }

    /// Internal: Publish message to a gossipsub topic
    async fn publish_to_topic(&self, topic: &str, msg_type: DexMessageType, payload: Vec<u8>) -> Result<()> {
        let message = DexNetworkMessage {
            message_type: msg_type,
            payload,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            node_id: self.local_node_id.clone(),
            signature: None, // TODO: Add signature for authenticity
        };

        let message_bytes = postcard::to_allocvec(&message)?;

        if let Some(ref tx) = self.network_cmd_tx {
            // Use the existing NetworkCommand::PublishDexEvent
            if let Err(e) = tx.send(super::NetworkCommand::PublishDexEvent {
                topic: topic.to_string(),
                message: message_bytes,
            }) {
                warn!("💱 [DEX P2P] Failed to send DEX event to network: {}", e);
                return Err(anyhow::anyhow!("Failed to send DEX event: {}", e));
            }
            debug!("💱 [DEX P2P] Published to topic: {}", topic);
        } else {
            warn!("💱 [DEX P2P] Network sender not connected - DEX event dropped");
            return Err(anyhow::anyhow!("Network sender not connected"));
        }

        Ok(())
    }

    /// Get publisher statistics
    pub fn get_stats(&self) -> DexPublisherStats {
        self.stats.clone()
    }

    /// Helper: Create a trade message from swap parameters
    pub fn create_trade_message(
        &self,
        trade_id: String,
        base_token: String,
        quote_token: String,
        price: u64,
        amount: u128,
        buyer: [u8; 32],
        seller: [u8; 32],
        buy_order_id: String,
        sell_order_id: String,
    ) -> TradeMessage {
        TradeMessage {
            trade_id,
            trading_pair: TradingPair {
                base: base_token,
                quote: quote_token,
            },
            buy_order_id,
            sell_order_id,
            price,
            amount,
            buyer,
            seller,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            executor_node: self.local_node_id.clone(),
        }
    }

    /// Helper: Create a liquidity pool message from pool state
    pub fn create_liquidity_message(
        pool_address: [u8; 32],
        token_a: String,
        token_b: String,
        reserve_a: u128,
        reserve_b: u128,
        total_liquidity: u128,
        fee_rate: u32,
    ) -> LiquidityPoolMessage {
        LiquidityPoolMessage {
            pool_address,
            token_a,
            token_b,
            reserve_a,
            reserve_b,
            total_liquidity,
            fee_rate,
            last_update: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Helper: Create a price update message
    pub fn create_price_message(
        base_token: String,
        quote_token: String,
        price: u64,
        volume_24h: u64,
        change_24h_bps: i64,
    ) -> PriceUpdateMessage {
        PriceUpdateMessage {
            trading_pair: TradingPair {
                base: base_token,
                quote: quote_token,
            },
            price,
            volume_24h,
            change_24h: change_24h_bps,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

/// Global DEX publisher instance (thread-safe)
pub type SharedDexPublisher = Arc<tokio::sync::RwLock<DexP2PPublisher>>;

/// Create a shared DEX publisher
pub fn create_shared_publisher(node_id: String) -> SharedDexPublisher {
    Arc::new(tokio::sync::RwLock::new(DexP2PPublisher::new(node_id)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_trade_message() {
        let publisher = DexP2PPublisher::new("test-node".to_string());
        let trade = publisher.create_trade_message(
            "trade-1".to_string(),
            "QUG".to_string(),
            "USDT".to_string(),
            100_000_000, // $1.00
            1_000_000_000, // 10 QUG
            [1u8; 32],
            [2u8; 32],
            "buy-1".to_string(),
            "sell-1".to_string(),
        );

        assert_eq!(trade.trading_pair.base, "QUG");
        assert_eq!(trade.trading_pair.quote, "USDT");
        assert_eq!(trade.price, 100_000_000);
        assert_eq!(trade.amount, 1_000_000_000);
    }

    #[test]
    fn test_create_liquidity_message() {
        let msg = DexP2PPublisher::create_liquidity_message(
            [0u8; 32],
            "QUG".to_string(),
            "USDT".to_string(),
            1_000_000_000_000, // 10,000 QUG
            1_000_000_000_000, // 10,000 USDT
            100_000_000_000,   // 1,000 LP tokens
            30,                // 0.30% fee
        );

        assert_eq!(msg.token_a, "QUG");
        assert_eq!(msg.token_b, "USDT");
        assert_eq!(msg.fee_rate, 30);
    }
}
