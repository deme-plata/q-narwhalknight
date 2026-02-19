use anyhow::{anyhow, Result};
use reqwest::Client;
use std::sync::Arc;

use crate::models::*;
use crate::wallet::Wallet;

/// Generic wrapper for all API responses: {"success":true,"data":{...}}
#[derive(serde::Deserialize)]
struct ApiWrapper<T> {
    #[allow(dead_code)]
    success: Option<bool>,
    data: Option<T>,
    error: Option<String>,
}

/// HTTP client that authenticates requests using the wallet's Ed25519 keys.
pub struct ApiClient {
    client: Client,
    base_url: String,
    wallet: Arc<Wallet>,
}

impl ApiClient {
    pub fn new(base_url: &str, wallet: Arc<Wallet>) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            wallet,
        }
    }

    /// GET request with wallet auth header, unwraps {"data":...} wrapper.
    async fn get_auth<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = format!("{}{}", self.base_url, path);
        let auth = self.wallet.auth_header(path);

        let resp = self
            .client
            .get(&url)
            .header("X-Wallet-Auth", &auth)
            .send()
            .await
            .map_err(|e| anyhow!("Request failed: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("HTTP {}: {}", status, body));
        }

        let wrapper: ApiWrapper<T> = resp
            .json()
            .await
            .map_err(|e| anyhow!("JSON parse error: {}", e))?;

        if let Some(err) = wrapper.error.filter(|e| !e.is_empty()) {
            return Err(anyhow!("API error: {}", err));
        }

        wrapper.data.ok_or_else(|| anyhow!("No data in response"))
    }

    /// GET request without auth (public endpoints), unwraps {"data":...}.
    async fn get_public<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = format!("{}{}", self.base_url, path);

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| anyhow!("Request failed: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("HTTP {}: {}", status, body));
        }

        let wrapper: ApiWrapper<T> = resp
            .json()
            .await
            .map_err(|e| anyhow!("JSON parse error: {}", e))?;

        if let Some(err) = wrapper.error.filter(|e| !e.is_empty()) {
            return Err(anyhow!("API error: {}", err));
        }

        wrapper.data.ok_or_else(|| anyhow!("No data in response"))
    }

    /// POST request with auth and JSON body, unwraps {"data":...}.
    async fn post_auth<T: serde::de::DeserializeOwned, B: serde::Serialize>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let url = format!("{}{}", self.base_url, path);
        let auth = self.wallet.auth_header(path);

        let resp = self
            .client
            .post(&url)
            .header("X-Wallet-Auth", &auth)
            .json(body)
            .send()
            .await
            .map_err(|e| anyhow!("Request failed: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("HTTP {}: {}", status, body));
        }

        let wrapper: ApiWrapper<T> = resp
            .json()
            .await
            .map_err(|e| anyhow!("JSON parse error: {}", e))?;

        if let Some(err) = wrapper.error.filter(|e| !e.is_empty()) {
            return Err(anyhow!("API error: {}", err));
        }

        wrapper.data.ok_or_else(|| anyhow!("No data in response"))
    }

    /// Fetch node sync status.
    pub async fn get_status(&self) -> Result<StatusResponse> {
        self.get_public("/api/v1/status").await
    }

    /// Fetch QUG wallet balance.
    pub async fn get_balance(&self) -> Result<BalanceResponse> {
        self.get_auth("/api/v1/wallet/balance").await
    }

    /// Fetch all custom token balances.
    pub async fn get_token_balances(&self) -> Result<MultiTokenBalanceResponse> {
        self.get_auth("/api/v1/multi-token-balance").await
    }

    /// Send a transaction.
    pub async fn send_transaction(
        &self,
        to: &str,
        amount: &str,
        memo: Option<String>,
    ) -> Result<serde_json::Value> {
        let tx = TransactionRequest {
            from: self.wallet.address().to_string(),
            to: to.to_string(),
            amount: amount.to_string(),
            fee: "0".to_string(),
            nonce: chrono::Utc::now().timestamp_millis() as u64,
            signature: String::new(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            memo,
        };

        let tx_data = format!("{}:{}:{}:{}", tx.from, tx.to, tx.amount, tx.nonce);
        let tx_hash = blake3::hash(tx_data.as_bytes());
        let signature = self.wallet.sign_transaction(tx_hash.as_bytes());

        let signed_tx = TransactionRequest {
            signature,
            ..tx
        };

        self.post_auth("/api/v1/transactions", &signed_tx).await
    }

    /// Fetch transaction history.
    pub async fn get_history(&self) -> Result<Vec<TransactionRecord>> {
        self.get_auth("/api/v1/transactions/history").await
    }

    /// Fetch current mining challenge.
    pub async fn get_mining_challenge(&self) -> Result<MiningChallenge> {
        self.get_public("/api/v1/mining/challenge").await
    }

    /// Submit a mining solution.
    pub async fn submit_mining_solution(
        &self,
        submission: &MiningSubmission,
    ) -> Result<serde_json::Value> {
        self.post_auth("/api/v1/mining/submit", submission).await
    }
}
