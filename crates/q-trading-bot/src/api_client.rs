/// Quillon DEX API client — connects to quillon.xyz REST endpoints.
///
/// Key endpoints used:
///   GET  /api/v1/dex/pools                 → list all liquidity pools
///   POST /api/v1/dex/swap/quote            → get AMM quote (no state change)
///   POST /api/v1/dex/swap/execute          → execute a swap on-chain
///   GET  /api/v1/defi/dex/status           → DEX health/status
///   GET  /api/v1/oracle/price/:token       → oracle price for a token
///
/// Amounts: all raw amounts are u128 with 24 decimal places.
///   display 1.0 QUG = raw 1_000_000_000_000_000_000_000_000 (10^24)

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;
use tracing::info;

pub const DECIMALS_24: u8 = 24;

/// Convert display amount to raw u128 with given decimals.
pub fn to_raw(display: f64, decimals: u8) -> u128 {
    let scale = 10u128.pow(decimals as u32);
    (display * scale as f64) as u128
}

/// Convert raw u128 to display amount.
pub fn to_display(raw: u128, decimals: u8) -> f64 {
    let scale = 10u128.pow(decimals as u32);
    raw as f64 / scale as f64
}

// ── API response wrapper ─────────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
pub struct ApiResponse<T> {
    pub success: Option<bool>,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T> ApiResponse<T> {
    pub fn into_data(self) -> Result<T> {
        if self.success.unwrap_or(true) {
            self.data.context("API returned success but no data")
        } else {
            Err(anyhow::anyhow!(
                "API error: {}",
                self.error.unwrap_or_else(|| "unknown".to_string())
            ))
        }
    }
}

// ── Pool types ───────────────────────────────────────────────────────────────

#[derive(Deserialize, Debug, Clone)]
pub struct PoolInfo {
    pub pool_id: String,
    pub token0: String,
    pub token1: String,
    pub reserve0: String,
    pub reserve1: String,
    pub fee_rate: Option<f64>,
    pub token0_decimals: Option<u8>,
    pub token1_decimals: Option<u8>,
}

impl PoolInfo {
    pub fn reserve0_raw(&self) -> u128 { self.reserve0.parse().unwrap_or(0) }
    pub fn reserve1_raw(&self) -> u128 { self.reserve1.parse().unwrap_or(0) }
    pub fn reserve0_display(&self) -> f64 {
        to_display(self.reserve0_raw(), self.token0_decimals.unwrap_or(24))
    }
    pub fn reserve1_display(&self) -> f64 {
        to_display(self.reserve1_raw(), self.token1_decimals.unwrap_or(24))
    }
    pub fn tvl_display(&self) -> f64 { self.reserve0_display() * 2.0 }
    pub fn ratio_0_over_1(&self) -> f64 {
        let r1 = self.reserve1_raw();
        if r1 == 0 { return 0.0; }
        self.reserve0_raw() as f64 / r1 as f64
    }
}

// ── Quote types ──────────────────────────────────────────────────────────────

#[derive(Serialize, Debug)]
pub struct QuoteRequest {
    pub token_in: String,
    pub token_out: String,
    pub amount_in: Option<String>,
    pub slippage_tolerance: Option<f64>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct SwapQuote {
    pub amount_in: String,
    pub amount_out: String,
    pub minimum_amount_out: String,
    pub price_impact: f64,
    pub gas_estimate: Option<u64>,
    pub execution_price: Option<f64>,
    pub valid_until: Option<u64>,
}

impl SwapQuote {
    pub fn amount_out_raw(&self) -> u128 { self.amount_out.parse().unwrap_or(0) }
    pub fn minimum_out_raw(&self) -> u128 { self.minimum_amount_out.parse().unwrap_or(0) }
}

// ── Execute types ────────────────────────────────────────────────────────────

#[derive(Serialize, Debug)]
pub struct SwapExecuteRequest {
    pub token_in: String,
    pub token_out: String,
    pub amount_in: String,
    pub minimum_amount_out: String,
    pub recipient: String,
    pub deadline: u64,
    pub signature: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct SwapResult {
    pub transaction_hash: String,
    pub status: String,
    pub amount_in: String,
    pub amount_out: String,
    pub gas_used: Option<u64>,
}

// ── Balance ──────────────────────────────────────────────────────────────────

#[derive(Deserialize, Debug, Clone)]
pub struct WalletBalance {
    pub qnk_balance: f64,
    pub custom_tokens: HashMap<String, f64>,
}

/// Legacy alias — engine.rs and wallet_manager.rs were written against ApiClient.
pub type ApiClient = QuillonClient;

// ── Client ───────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct QuillonClient {
    client: Client,
    pub base_url: String,
}

impl QuillonClient {
    pub fn new(base_url: String) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .user_agent("q-water-bot/1.0")
            .build()
            .expect("failed to build HTTP client");
        QuillonClient { client, base_url }
    }

    pub async fn list_pools(&self) -> Result<Vec<PoolInfo>> {
        let url = format!("{}/api/v1/dex/pools", self.base_url);
        debug!("GET {}", url);
        let resp: serde_json::Value = self.client.get(&url).send().await?.json().await?;
        // Handle {success, data: [...]}, {pools: [...]}, or plain array
        if let Some(arr) = resp.as_array() {
            return Ok(serde_json::from_value(serde_json::Value::Array(arr.clone())).unwrap_or_default());
        }
        if let Some(data) = resp.get("data") {
            if let Some(arr) = data.as_array() {
                return Ok(serde_json::from_value(serde_json::Value::Array(arr.clone())).unwrap_or_default());
            }
            if let Some(pools) = data.get("pools").and_then(|v| v.as_array()) {
                return Ok(serde_json::from_value(serde_json::Value::Array(pools.clone())).unwrap_or_default());
            }
        }
        Ok(vec![])
    }

    pub async fn find_pool(&self, token_in: &str, token_out: &str) -> Result<Option<PoolInfo>> {
        let pools = self.list_pools().await?;
        let ti = token_in.to_uppercase();
        let to_ = token_out.to_uppercase();
        let best = pools.into_iter()
            .filter(|p| {
                let p0 = p.token0.to_uppercase();
                let p1 = p.token1.to_uppercase();
                (p0 == ti && p1 == to_) || (p0 == to_ && p1 == ti)
            })
            .max_by(|a, b| {
                let ka = a.reserve0_raw() as f64 * a.reserve1_raw() as f64;
                let kb = b.reserve0_raw() as f64 * b.reserve1_raw() as f64;
                ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
            });
        Ok(best)
    }

    pub async fn get_quote(
        &self,
        token_in: &str,
        token_out: &str,
        amount_in_raw: u128,
        slippage: f64,
    ) -> Result<SwapQuote> {
        let url = format!("{}/api/v1/dex/swap/quote", self.base_url);
        let body = QuoteRequest {
            token_in: token_in.to_string(),
            token_out: token_out.to_string(),
            amount_in: Some(amount_in_raw.to_string()),
            slippage_tolerance: Some(slippage),
        };
        let resp: ApiResponse<SwapQuote> =
            self.client.post(&url).json(&body).send().await?.json().await?;
        resp.into_data()
    }

    pub async fn execute_swap(
        &self,
        token_in: &str,
        token_out: &str,
        amount_in_raw: u128,
        minimum_out_raw: u128,
        wallet_address: &str,
    ) -> Result<SwapResult> {
        let deadline = chrono::Utc::now().timestamp() as u64 + 300;
        let url = format!("{}/api/v1/dex/swap/execute", self.base_url);
        let body = SwapExecuteRequest {
            token_in: token_in.to_string(),
            token_out: token_out.to_string(),
            amount_in: amount_in_raw.to_string(),
            minimum_amount_out: minimum_out_raw.to_string(),
            recipient: wallet_address.to_string(),
            deadline,
            signature: format!("water-bot-{}", chrono::Utc::now().timestamp_millis()),
        };
        debug!("POST {} | amount_in={} min_out={}", url, amount_in_raw, minimum_out_raw);
        let resp: ApiResponse<SwapResult> =
            self.client.post(&url).json(&body).send().await?.json().await?;
        let result = resp.into_data()?;
        let hash_preview = &result.transaction_hash[..result.transaction_hash.len().min(18)];
        info!("✅ Swap: {} → {} | tx={}… | {}", token_in, token_out, hash_preview, result.status);
        Ok(result)
    }

    pub async fn get_price(&self, token: &str) -> Result<f64> {
        let url = format!("{}/api/v1/oracle/price/{}", self.base_url, token);
        let resp: serde_json::Value = self.client.get(&url).send().await?.json().await?;
        if let Some(p) = resp.get("price_usd").and_then(|v| v.as_f64()) { return Ok(p); }
        if let Some(p) = resp.pointer("/data/price_usd").and_then(|v| v.as_f64()) { return Ok(p); }
        Ok(0.0)
    }

    pub async fn dex_status(&self) -> bool {
        let url = format!("{}/api/v1/defi/dex/status", self.base_url);
        self.client.get(&url).send().await.map(|r| r.status().is_success()).unwrap_or(false)
    }

    /// Stub for legacy callers.
    pub async fn get_all_balances(&self) -> Result<Vec<(String, WalletBalance)>> {
        Ok(vec![])
    }

    // ════════════════════════════════════════════════════════════════════
    // QSHARE-1 Phase 1.1 — X-Wallet-Auth signed endpoints for QCREDIT-DCA
    // ════════════════════════════════════════════════════════════════════

    /// GET /api/v1/wallets/<addr>/balance with X-Wallet-Auth signature.
    /// Returns the QUG balance in display units (e.g. 0.05).
    /// Per AFL-1 §3 and AGENT.md §2 — signed challenge = SHA3-256(addr ‖ ts_le ‖ path).
    pub async fn get_balance_qug_signed(
        &self,
        wallet: &str,
        signer: &crate::wallet_auth::AgentWallet,
    ) -> Result<f64> {
        let path = format!("/api/v1/wallets/{}/balance", wallet);
        let auth_header = signer.sign_request(&path)
            .context("sign get_balance request")?;
        let url = format!("{}{}", self.base_url, path);
        debug!("GET {} (signed)", url);
        let resp = self.client
            .get(&url)
            .header("X-Wallet-Auth", auth_header)
            .send()
            .await
            .context("balance request")?;
        let status = resp.status();
        let body: serde_json::Value = resp.json().await.context("balance json")?;
        if !status.is_success() {
            return Err(anyhow::anyhow!(
                "balance fetch failed: status={} body={}",
                status, body
            ));
        }
        // Response shape: {success, data: {qug_balance: "<raw u128 str>" | float, ...}}
        let data = body.get("data").unwrap_or(&body);
        // Try multiple field names + numeric representations the API may use
        let qug = data.get("qug_balance")
            .or_else(|| data.get("balance"))
            .or_else(|| data.get("qug"));
        match qug {
            Some(serde_json::Value::String(s)) => {
                let raw: u128 = s.parse().context("balance str→u128")?;
                Ok(to_display(raw, DECIMALS_24))
            }
            Some(serde_json::Value::Number(n)) => {
                if let Some(f) = n.as_f64() { Ok(f) }
                else if let Some(u) = n.as_u64() { Ok(to_display(u as u128, DECIMALS_24)) }
                else { Ok(0.0) }
            }
            _ => Ok(0.0),
        }
    }

    /// POST /api/v1/qcredit/lock — lock QUG into a QCREDIT yield tier.
    /// Returns the position_id as a String.
    /// Endpoint contract: crates/q-api-server/src/qcredit_api.rs::LockRequest
    pub async fn qcredit_lock_signed(
        &self,
        wallet: &str,
        amount_qug: f64,
        tier: &str,
        signer: &crate::wallet_auth::AgentWallet,
    ) -> Result<String> {
        let path = "/api/v1/qcredit/lock";
        let auth_header = signer.sign_request(path).context("sign qcredit_lock")?;
        let url = format!("{}{}", self.base_url, path);
        let body = serde_json::json!({
            "wallet": wallet,
            "amount": format!("{}", amount_qug),
            "tier": tier,
        });
        debug!("POST {} | wallet={} amount={} tier={}", url, wallet, amount_qug, tier);
        let resp = self.client
            .post(&url)
            .header("X-Wallet-Auth", auth_header)
            .json(&body)
            .send()
            .await
            .context("qcredit_lock request")?;
        let status = resp.status();
        let json: serde_json::Value = resp.json().await.context("qcredit_lock json")?;
        if !status.is_success() {
            return Err(anyhow::anyhow!(
                "qcredit_lock failed: status={} body={}",
                status, json
            ));
        }
        // Response shape: {success, data: {position_id: "...", ...}}
        let position_id = json.pointer("/data/position_id")
            .or_else(|| json.get("position_id"))
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| format!("unknown-{}", chrono::Utc::now().timestamp_millis()));
        info!("✅ QCREDIT lock: {:.6} QUG into {} tier (position={})", amount_qug, tier, position_id);
        Ok(position_id)
    }
}
