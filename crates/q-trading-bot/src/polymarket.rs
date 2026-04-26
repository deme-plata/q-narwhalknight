/// Polymarket REST API client.
///
/// Fetches BTC-related prediction market data:
///   - gamma-api.polymarket.com  → market metadata (title, strike, expiry, tags)
///   - clob.polymarket.com       → live YES/NO bid-ask prices (the CLOB)
///
/// Polymarket prices are in the [0, 1] range where:
///   price = 0.65 means 65% probability the outcome resolves YES.
///   YES shares + NO shares always sum to ~1.00 USDC.
///
/// BTC markets of interest:
///   "Will BTC close above $X on [date]?" → strike K, expiry T
///   The YES mid-price is our Polymarket probability estimate.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;
use tracing::debug;

const GAMMA_API: &str = "https://gamma-api.polymarket.com";
const CLOB_API:  &str = "https://clob.polymarket.com";

// ── Gamma API types ──────────────────────────────────────────────────────────

#[derive(Deserialize, Debug, Clone)]
pub struct PolyMarket {
    pub id: String,
    #[serde(rename = "conditionId")]
    pub condition_id: Option<String>,
    pub question: String,
    pub description: Option<String>,
    pub slug: Option<String>,
    /// End date / expiry of the market (ISO 8601)
    #[serde(rename = "endDate")]
    pub end_date: Option<String>,
    pub active: Option<bool>,
    pub closed: Option<bool>,
    pub volume: Option<f64>,
    /// Current best YES price from gamma metadata
    #[serde(rename = "bestBid")]
    pub best_bid: Option<f64>,
    #[serde(rename = "bestAsk")]
    pub best_ask: Option<f64>,
    /// Tags: ["bitcoin", "crypto", ...]
    pub tags: Option<Vec<serde_json::Value>>,
}

impl PolyMarket {
    /// YES mid-price from gamma metadata (0–1).
    pub fn yes_mid(&self) -> Option<f64> {
        match (self.best_bid, self.best_ask) {
            (Some(b), Some(a)) if b > 0.0 && a > 0.0 => Some((b + a) / 2.0),
            (Some(b), _) if b > 0.0 => Some(b),
            (_, Some(a)) if a > 0.0 => Some(a),
            _ => None,
        }
    }

    /// Rough strike parse: find a dollar amount in the question like "$100,000" or "$80000".
    pub fn parse_strike(&self) -> Option<f64> {
        parse_dollar_amount(&self.question)
    }

    /// Days until market closes from now.
    pub fn days_to_expiry(&self) -> Option<f64> {
        let end = self.end_date.as_deref()?;
        let dt = chrono::DateTime::parse_from_rfc3339(end)
            .or_else(|_| chrono::DateTime::parse_from_str(end, "%Y-%m-%dT%H:%M:%S%.3fZ"))
            .ok()?;
        let now = chrono::Utc::now();
        let secs = (dt.with_timezone(&chrono::Utc) - now).num_seconds();
        if secs <= 0 { None } else { Some(secs as f64 / 86400.0) }
    }
}

// ── CLOB types ───────────────────────────────────────────────────────────────

#[derive(Deserialize, Debug, Clone)]
pub struct ClobMarket {
    pub condition_id: String,
    pub tokens: Vec<ClobToken>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ClobToken {
    pub token_id: String,
    pub outcome: String,  // "Yes" or "No"
    pub price: Option<f64>,
}

impl ClobMarket {
    pub fn yes_price(&self) -> Option<f64> {
        self.tokens.iter()
            .find(|t| t.outcome.to_lowercase() == "yes")
            .and_then(|t| t.price)
    }
}

// ── Parsed BTC market ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BtcPolymarketOpportunity {
    pub condition_id: String,
    pub question: String,
    /// Strike price in USD (parsed from question)
    pub strike_usd: f64,
    /// Days until expiry
    pub days_to_expiry: f64,
    /// Polymarket YES probability (0–1)
    pub poly_prob: f64,
    /// 24h volume in USDC
    pub volume_usdc: f64,
}

// ── Client ───────────────────────────────────────────────────────────────────

pub struct PolymarketClient {
    client: Client,
}

impl PolymarketClient {
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(15))
            .user_agent("dark-knight-bot/1.0")
            .build()
            .expect("reqwest client");
        PolymarketClient { client }
    }

    /// Fetch active BTC markets from the Gamma API.
    pub async fn fetch_btc_markets(&self) -> Result<Vec<PolyMarket>> {
        // tag=bitcoin returns BTC-related markets
        let url = format!("{}/markets?tag=bitcoin&active=true&closed=false&limit=50", GAMMA_API);
        debug!("Polymarket GET {}", url);
        let resp: serde_json::Value = self.client.get(&url).send().await?.json().await?;

        // Response is either a plain array or {markets: [...]}
        let arr = if let Some(a) = resp.as_array() {
            a.clone()
        } else if let Some(a) = resp.get("markets").and_then(|v| v.as_array()) {
            a.clone()
        } else {
            vec![]
        };

        let markets: Vec<PolyMarket> = arr.into_iter()
            .filter_map(|v| serde_json::from_value(v).ok())
            .collect();
        Ok(markets)
    }

    /// Fetch live YES price from the CLOB for a specific condition_id.
    pub async fn fetch_clob_price(&self, condition_id: &str) -> Result<f64> {
        let url = format!("{}/markets/{}", CLOB_API, condition_id);
        debug!("Polymarket CLOB GET {}", url);
        let resp: serde_json::Value = self.client.get(&url).send().await?
            .json().await?;

        // CLOB response: {tokens: [{outcome: "Yes", price: 0.65}, ...]}
        if let Some(tokens) = resp.get("tokens").and_then(|v| v.as_array()) {
            for tok in tokens {
                let outcome = tok.get("outcome").and_then(|v| v.as_str()).unwrap_or("");
                if outcome.to_lowercase() == "yes" {
                    if let Some(p) = tok.get("price").and_then(|v| v.as_f64()) {
                        return Ok(p);
                    }
                }
            }
        }

        // Fallback: try mid-price from bid/ask
        if let (Some(b), Some(a)) = (
            resp.pointer("/bestBid").and_then(|v| v.as_f64()),
            resp.pointer("/bestAsk").and_then(|v| v.as_f64()),
        ) {
            return Ok((b + a) / 2.0);
        }

        anyhow::bail!("Cannot parse YES price from CLOB response for {}", condition_id)
    }

    /// Find all BTC price-target markets that have a parseable strike and expiry.
    pub async fn find_btc_opportunities(&self) -> Result<Vec<BtcPolymarketOpportunity>> {
        let markets = self.fetch_btc_markets().await
            .context("Fetching Polymarket BTC markets")?;

        let mut opps = Vec::new();
        for m in markets {
            let strike = match m.parse_strike() {
                Some(s) => s,
                None => continue,
            };
            let dte = match m.days_to_expiry() {
                Some(d) if d > 0.5 => d,
                _ => continue, // expired or too close
            };
            // Prefer CLOB price; fall back to gamma metadata mid
            let cid = match &m.condition_id {
                Some(c) if !c.is_empty() => c.clone(),
                _ => m.id.clone(),
            };
            let poly_prob = if let Ok(p) = self.fetch_clob_price(&cid).await {
                p
            } else {
                match m.yes_mid() {
                    Some(p) => p,
                    None => continue,
                }
            };

            if poly_prob <= 0.01 || poly_prob >= 0.99 { continue; } // degenerate

            opps.push(BtcPolymarketOpportunity {
                condition_id: cid,
                question: m.question.clone(),
                strike_usd: strike,
                days_to_expiry: dte,
                poly_prob,
                volume_usdc: m.volume.unwrap_or(0.0),
            });
        }

        // Sort by volume descending (most liquid first)
        opps.sort_by(|a, b| b.volume_usdc.partial_cmp(&a.volume_usdc).unwrap_or(std::cmp::Ordering::Equal));
        Ok(opps)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Parse first dollar amount from text, e.g. "$100,000" → 100000.0
fn parse_dollar_amount(text: &str) -> Option<f64> {
    let mut in_num = false;
    let mut buf = String::new();
    let mut found_dollar = false;

    for ch in text.chars() {
        if ch == '$' {
            found_dollar = true;
            in_num = true;
            buf.clear();
        } else if in_num && (ch.is_ascii_digit() || ch == ',' || ch == '.') {
            buf.push(ch);
        } else if in_num {
            in_num = false;
            if found_dollar && !buf.is_empty() {
                break;
            }
            buf.clear();
        }
    }

    if !found_dollar || buf.is_empty() { return None; }
    buf.retain(|c| c != ',');
    buf.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dollar() {
        assert_eq!(parse_dollar_amount("Will BTC close above $100,000?"), Some(100000.0));
        assert_eq!(parse_dollar_amount("BTC above $80000 by Dec"), Some(80000.0));
        assert_eq!(parse_dollar_amount("BTC price $150,000 target"), Some(150000.0));
        assert_eq!(parse_dollar_amount("no dollar sign"), None);
    }
}
