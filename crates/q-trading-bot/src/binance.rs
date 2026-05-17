/// Binance Futures (USDM) REST + WebSocket client.
///
/// Authentication: HMAC-SHA256 over the query string with your secret key.
/// All amounts in BTC (base) or USDT (quote).
///
/// Key endpoints:
///   GET  /fapi/v1/ticker/price              → current mark price
///   GET  /fapi/v1/klines                    → candlestick OHLCV
///   GET  /fapi/v2/account                   → balances + positions
///   POST /fapi/v1/order                     → place order
///   DELETE /fapi/v1/order                   → cancel order
///   GET  /fapi/v1/positionRisk              → open positions + PnL
///   POST /fapi/v1/leverage                  → set leverage
///
/// Rate limits: 1200 weight/min REST, 10 orders/sec per symbol.
///
/// IMPORTANT: Binance Futures requires testnet for paper trading.
///   Testnet: https://testnet.binancefuture.com
///   Mainnet: https://fapi.binance.com

use anyhow::{Context, Result};
use chrono::Utc;
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::time::Duration;
use tracing::{debug, info};

type HmacSha256 = Hmac<Sha256>;

const MAINNET: &str = "https://fapi.binance.com";
const TESTNET: &str = "https://testnet.binancefuture.com";

// ── Config ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BinanceConfig {
    pub api_key: String,
    pub secret_key: String,
    /// Use testnet (paper trading). ALWAYS start with testnet!
    pub testnet: bool,
    /// Max leverage (1–125). Recommend ≤5 for systematic strategies.
    pub leverage: u32,
    /// Fraction of available USDT margin to use per trade (0.0–1.0).
    pub position_size_fraction: f64,
    /// Minimum notional value per order (Binance minimum is $5 for BTC futures).
    pub min_notional_usdt: f64,
}

impl Default for BinanceConfig {
    fn default() -> Self {
        BinanceConfig {
            api_key: String::new(),
            secret_key: String::new(),
            testnet: true, // ALWAYS paper trade first
            leverage: 3,
            position_size_fraction: 0.05, // 5% of margin per trade
            min_notional_usdt: 10.0,
        }
    }
}

// ── Response types ───────────────────────────────────────────────────────────

#[derive(Deserialize, Debug, Clone)]
pub struct MarkPrice {
    pub symbol: String,
    pub price: String,
}

impl MarkPrice {
    pub fn price_f64(&self) -> f64 { self.price.parse().unwrap_or(0.0) }
}

#[derive(Deserialize, Debug, Clone)]
pub struct Kline {
    pub open_time: u64,
    pub open: String,
    pub high: String,
    pub low: String,
    pub close: String,
    pub volume: String,
    pub close_time: u64,
}

impl Kline {
    pub fn close_f64(&self) -> f64 { self.close.parse().unwrap_or(0.0) }
    pub fn high_f64(&self) -> f64 { self.high.parse().unwrap_or(0.0) }
    pub fn low_f64(&self) -> f64 { self.low.parse().unwrap_or(0.0) }
    pub fn volume_f64(&self) -> f64 { self.volume.parse().unwrap_or(0.0) }
}

#[derive(Deserialize, Debug, Clone)]
pub struct AccountBalance {
    pub asset: String,
    #[serde(rename = "availableBalance")]
    pub available_balance: String,
    #[serde(rename = "walletBalance")]
    pub wallet_balance: String,
}

impl AccountBalance {
    pub fn available_f64(&self) -> f64 { self.available_balance.parse().unwrap_or(0.0) }
}

#[derive(Deserialize, Debug, Clone)]
pub struct Position {
    pub symbol: String,
    #[serde(rename = "positionAmt")]
    pub position_amt: String,
    #[serde(rename = "entryPrice")]
    pub entry_price: String,
    #[serde(rename = "unRealizedProfit")]
    pub unrealized_profit: String,
    #[serde(rename = "leverage")]
    pub leverage: String,
    #[serde(rename = "positionSide")]
    pub position_side: String,
}

impl Position {
    pub fn size_f64(&self) -> f64 { self.position_amt.parse().unwrap_or(0.0) }
    pub fn entry_f64(&self) -> f64 { self.entry_price.parse().unwrap_or(0.0) }
    pub fn pnl_f64(&self) -> f64 { self.unrealized_profit.parse().unwrap_or(0.0) }
    pub fn is_long(&self) -> bool { self.size_f64() > 0.0 }
    pub fn is_short(&self) -> bool { self.size_f64() < 0.0 }
    pub fn is_flat(&self) -> bool { self.size_f64().abs() < 1e-8 }
}

#[derive(Deserialize, Debug, Clone)]
pub struct OrderResponse {
    #[serde(rename = "orderId")]
    pub order_id: u64,
    pub symbol: String,
    pub side: String,
    pub r#type: String,
    #[serde(rename = "origQty")]
    pub orig_qty: String,
    #[serde(rename = "executedQty")]
    pub executed_qty: String,
    pub status: String,
    pub price: String,
    #[serde(rename = "avgPrice")]
    pub avg_price: Option<String>,
}

impl OrderResponse {
    pub fn filled_qty(&self) -> f64 { self.executed_qty.parse().unwrap_or(0.0) }
    pub fn avg_fill_price(&self) -> f64 {
        self.avg_price.as_deref().and_then(|s| s.parse().ok()).unwrap_or(0.0)
    }
}

// ── Order side / type ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side { Buy, Sell }

impl Side {
    pub fn as_str(self) -> &'static str {
        match self { Side::Buy => "BUY", Side::Sell => "SELL" }
    }
    pub fn opposite(self) -> Self {
        match self { Side::Buy => Side::Sell, Side::Sell => Side::Buy }
    }
}

// ── Client ───────────────────────────────────────────────────────────────────

pub struct BinanceFutures {
    client: Client,
    cfg: BinanceConfig,
    base_url: String,
}

impl BinanceFutures {
    pub fn new(cfg: BinanceConfig) -> Self {
        let base_url = if cfg.testnet { TESTNET } else { MAINNET }.to_string();
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("reqwest client");
        BinanceFutures { client, cfg, base_url }
    }

    // ── Auth ─────────────────────────────────────────────────────────────────

    fn sign(&self, query: &str) -> String {
        let mut mac = HmacSha256::new_from_slice(self.cfg.secret_key.as_bytes())
            .expect("HMAC key");
        mac.update(query.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    fn timestamp_ms() -> u64 {
        Utc::now().timestamp_millis() as u64
    }

    fn signed_query(&self, params: &[(&str, &str)]) -> String {
        let mut q: Vec<String> = params.iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        let ts = Self::timestamp_ms().to_string();
        q.push(format!("timestamp={}", ts));
        let qs = q.join("&");
        let sig = self.sign(&qs);
        format!("{}&signature={}", qs, sig)
    }

    // ── Public endpoints ──────────────────────────────────────────────────────

    /// Current mark price for a symbol (e.g. "BTCUSDT").
    pub async fn mark_price(&self, symbol: &str) -> Result<f64> {
        let url = format!("{}/fapi/v1/ticker/price?symbol={}", self.base_url, symbol);
        debug!("Binance GET {}", url);
        let resp: serde_json::Value = self.client.get(&url).send().await?.json().await?;
        resp.get("price").and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .context("Cannot parse mark price")
    }

    /// Fetch recent klines (candlesticks).
    /// interval: "1m", "5m", "1h", "1d", etc.
    pub async fn klines(&self, symbol: &str, interval: &str, limit: u32) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/fapi/v1/klines?symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );
        debug!("Binance GET {}", url);
        let resp: Vec<serde_json::Value> = self.client.get(&url).send().await?.json().await?;

        let klines = resp.into_iter().filter_map(|row| {
            let arr = row.as_array()?;
            Some(Kline {
                open_time: arr.get(0)?.as_u64()?,
                open:       arr.get(1)?.as_str()?.to_string(),
                high:       arr.get(2)?.as_str()?.to_string(),
                low:        arr.get(3)?.as_str()?.to_string(),
                close:      arr.get(4)?.as_str()?.to_string(),
                volume:     arr.get(5)?.as_str()?.to_string(),
                close_time: arr.get(6)?.as_u64()?,
            })
        }).collect();
        Ok(klines)
    }

    /// 30-day realised volatility (annualised) from daily closes.
    pub async fn historical_volatility(&self, symbol: &str) -> Result<f64> {
        let klines = self.klines(symbol, "1d", 31).await?;
        if klines.len() < 2 { anyhow::bail!("Not enough klines for vol calculation"); }

        let closes: Vec<f64> = klines.iter().map(|k| k.close_f64()).filter(|&c| c > 0.0).collect();
        let log_returns: Vec<f64> = closes.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        let n = log_returns.len() as f64;
        let mean = log_returns.iter().sum::<f64>() / n;
        let variance = log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let daily_vol = variance.sqrt();
        Ok(daily_vol * (365.0_f64).sqrt()) // annualise
    }

    // ── Authenticated endpoints ───────────────────────────────────────────────

    /// Current USDT balance available for futures.
    pub async fn usdt_balance(&self) -> Result<f64> {
        let qs = self.signed_query(&[("recvWindow", "5000")]);
        let url = format!("{}/fapi/v2/account?{}", self.base_url, qs);
        let resp: serde_json::Value = self.client.get(&url)
            .header("X-MBX-APIKEY", &self.cfg.api_key)
            .send().await?.json().await?;

        let assets = resp.get("assets")
            .and_then(|v| v.as_array())
            .context("No assets in account response")?;

        for asset in assets {
            if asset.get("asset").and_then(|v| v.as_str()) == Some("USDT") {
                if let Some(bal) = asset.get("availableBalance").and_then(|v| v.as_str()) {
                    return bal.parse::<f64>().context("Parse USDT balance");
                }
            }
        }
        Ok(0.0)
    }

    /// Current position for a symbol.
    pub async fn position(&self, symbol: &str) -> Result<Position> {
        let sym_param = symbol.to_string();
        let qs = self.signed_query(&[("symbol", &sym_param), ("recvWindow", "5000")]);
        let url = format!("{}/fapi/v2/positionRisk?{}", self.base_url, qs);
        let resp: Vec<serde_json::Value> = self.client.get(&url)
            .header("X-MBX-APIKEY", &self.cfg.api_key)
            .send().await?.json().await?;

        resp.into_iter()
            .filter_map(|v| serde_json::from_value::<Position>(v).ok())
            .find(|p| p.symbol == symbol)
            .context("Position not found")
    }

    /// Set leverage for a symbol (idempotent).
    pub async fn set_leverage(&self, symbol: &str, leverage: u32) -> Result<()> {
        let lev = leverage.to_string();
        let qs = self.signed_query(&[("symbol", symbol), ("leverage", &lev)]);
        let url = format!("{}/fapi/v1/leverage", self.base_url);
        self.client.post(&url)
            .header("X-MBX-APIKEY", &self.cfg.api_key)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(qs)
            .send().await?;
        Ok(())
    }

    /// Place a MARKET order. Returns the order response.
    pub async fn market_order(
        &self,
        symbol: &str,
        side: Side,
        qty_btc: f64,
    ) -> Result<OrderResponse> {
        let qty_str = format!("{:.3}", qty_btc); // 3 decimal places for BTC
        let qs = self.signed_query(&[
            ("symbol",   symbol),
            ("side",     side.as_str()),
            ("type",     "MARKET"),
            ("quantity", &qty_str),
            ("recvWindow", "5000"),
        ]);
        let url = format!("{}/fapi/v1/order", self.base_url);
        let resp = self.client.post(&url)
            .header("X-MBX-APIKEY", &self.cfg.api_key)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(qs)
            .send().await?;

        let status = resp.status();
        let body: serde_json::Value = resp.json().await?;
        if !status.is_success() {
            anyhow::bail!("Binance order error {}: {}", status, body);
        }
        let order: OrderResponse = serde_json::from_value(body)?;
        info!("✅ Binance {} {} {:.3} BTC | id={} status={}",
            order.side, symbol, order.filled_qty(), order.order_id, order.status);
        Ok(order)
    }

    /// Close an existing position (place market order in opposite direction).
    pub async fn close_position(&self, symbol: &str, position: &Position) -> Result<OrderResponse> {
        let size = position.size_f64().abs();
        let side = if position.is_long() { Side::Sell } else { Side::Buy };
        info!("🔒 Closing {} {} position ({:.3} BTC)", symbol, if position.is_long() { "LONG" } else { "SHORT" }, size);
        self.market_order(symbol, side, size).await
    }

    /// Compute BTC quantity from USDT notional.
    pub async fn usdt_to_btc_qty(&self, symbol: &str, usdt: f64) -> Result<f64> {
        let price = self.mark_price(symbol).await?;
        Ok(usdt / price)
    }
}
