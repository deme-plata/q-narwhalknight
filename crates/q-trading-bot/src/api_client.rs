/// API client for Q-NarwhalKnight node
use anyhow::{Context, Result};
use reqwest::Client;
use rust_decimal::Decimal;
use std::collections::HashMap;
use crate::types::*;

pub struct ApiClient {
    client: Client,
    base_url: String,
}

impl ApiClient {
    pub fn new(base_url: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
        }
    }

    /// Get wallet balance
    pub async fn get_balance(&self, wallet_id: &str) -> Result<WalletBalance> {
        let url = format!("{}/wallet/{}/balance", self.base_url, wallet_id);
        let response = self.client.get(&url).send().await?;
        
        #[derive(serde::Deserialize)]
        struct BalanceResponse {
            balance: u64,
            custom_tokens: Option<HashMap<String, u64>>,
        }
        
        let balance_resp: BalanceResponse = response.json().await?;
        
        Ok(WalletBalance {
            wallet_id: wallet_id.to_string(),
            qnk_balance: Decimal::from(balance_resp.balance),
            custom_tokens: balance_resp.custom_tokens.unwrap_or_default()
                .into_iter()
                .map(|(k, v)| (k, Decimal::from(v)))
                .collect(),
        })
    }

    /// Get all balances
    pub async fn get_all_balances(&self) -> Result<HashMap<String, WalletBalance>> {
        let url = format!("{}/wallets/balances", self.base_url);
        let response = self.client.get(&url).send().await?;
        let balances: HashMap<String, WalletBalance> = response.json().await?;
        Ok(balances)
    }

    /// Send transaction
    pub async fn send_transaction(&self, from: &str, to: &str, amount: Decimal, token: Option<&str>) -> Result<TradeResult> {
        let url = format!("{}/transaction/send", self.base_url);
        
        let body = serde_json::json!({
            "from": from,
            "to": to,
            "amount": amount.to_string(),
            "token": token,
        });
        
        let response = self.client.post(&url).json(&body).send().await?;
        let result: TradeResult = response.json().await?;
        Ok(result)
    }

    /// Get ticker data
    pub async fn get_ticker(&self, pair: &TradingPair) -> Result<Ticker> {
        let url = format!("{}/market/ticker/{}", self.base_url, pair.symbol());
        let response = self.client.get(&url).send().await?;
        let ticker: Ticker = response.json().await?;
        Ok(ticker)
    }

    /// Get order book
    pub async fn get_order_book(&self, pair: &TradingPair, depth: usize) -> Result<OrderBook> {
        let url = format!("{}/market/orderbook/{}?depth={}", self.base_url, pair.symbol(), depth);
        let response = self.client.get(&url).send().await?;
        let order_book: OrderBook = response.json().await?;
        Ok(order_book)
    }

    /// Place order
    pub async fn place_order(&self, order: &Order) -> Result<Order> {
        let url = format!("{}/market/order", self.base_url);
        let response = self.client.post(&url).json(order).send().await?;
        let placed_order: Order = response.json().await?;
        Ok(placed_order)
    }

    /// Cancel order
    pub async fn cancel_order(&self, order_id: &str) -> Result<()> {
        let url = format!("{}/market/order/{}/cancel", self.base_url, order_id);
        self.client.delete(&url).send().await?;
        Ok(())
    }
}
