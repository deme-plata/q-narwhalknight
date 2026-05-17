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

/// Authentication mode for the API client.
enum AuthMode {
    /// Wallet-based auth using Ed25519 signature (X-Wallet-Auth header).
    Wallet(Arc<Wallet>),
    /// OAuth2 Bearer token auth (Authorization: Bearer header).
    Bearer { token: String, address: String },
}

/// HTTP client that authenticates requests using either wallet signatures or OAuth2 Bearer tokens.
pub struct ApiClient {
    client: Client,
    base_url: String,
    auth: AuthMode,
}

impl ApiClient {
    /// Build a reqwest client with proper connection management.
    /// v8.5.2: Matches standalone miner — TCP keepalive, connection pooling, idle timeouts.
    fn build_client() -> Client {
        Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(15))
            .pool_max_idle_per_host(2)
            .pool_idle_timeout(std::time::Duration::from_secs(30))
            .tcp_keepalive(std::time::Duration::from_secs(15))
            .build()
            .unwrap_or_else(|_| Client::new())
    }

    /// Create an API client using wallet-based Ed25519 signature authentication.
    pub fn new(base_url: &str, wallet: Arc<Wallet>) -> Self {
        Self {
            client: Self::build_client(),
            base_url: base_url.trim_end_matches('/').to_string(),
            auth: AuthMode::Wallet(wallet),
        }
    }

    /// Create an API client using OAuth2 Bearer token authentication.
    /// This mode supports balance queries, history, and send transactions
    /// without requiring a local private key.
    pub fn from_bearer(base_url: &str, token: String, address: String) -> Self {
        Self {
            client: Self::build_client(),
            base_url: base_url.trim_end_matches('/').to_string(),
            auth: AuthMode::Bearer { token, address },
        }
    }

    /// Get the wallet address for this client.
    pub fn address(&self) -> &str {
        match &self.auth {
            AuthMode::Wallet(w) => w.address(),
            AuthMode::Bearer { address, .. } => address,
        }
    }

    /// Get the base URL for SSE/fallback connections.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns true if this client has a local wallet (can sign locally).
    pub fn has_wallet(&self) -> bool {
        matches!(&self.auth, AuthMode::Wallet(_))
    }

    /// Get wallet address as hex (without qnk prefix), for SSE query parameter.
    pub fn address_hex(&self) -> &str {
        let addr = self.address();
        if addr.starts_with("qnk") { &addr[3..] } else { addr }
    }

    /// GET request with auth header, unwraps {"data":...} wrapper.
    async fn get_auth<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = format!("{}{}", self.base_url, path);

        let req = self.client.get(&url);
        let req = match &self.auth {
            AuthMode::Wallet(wallet) => req.header("X-Wallet-Auth", wallet.auth_header(path)),
            AuthMode::Bearer { token, .. } => {
                req.header("Authorization", format!("Bearer {}", token))
            }
        };

        let resp = req
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

    /// GET request without auth, returns raw JSON Value (for endpoints with non-standard wrappers).
    pub async fn get_public_raw(&self, url: &str) -> Result<serde_json::Value> {
        let resp = self.client.get(url).send().await
            .map_err(|e| anyhow!("Request failed: {}", e))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("HTTP {}: {}", status, body));
        }
        resp.json().await.map_err(|e| anyhow!("JSON parse error: {}", e))
    }

    /// POST request with auth and JSON body, unwraps {"data":...}.
    async fn post_auth<T: serde::de::DeserializeOwned, B: serde::Serialize>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let url = format!("{}{}", self.base_url, path);

        let req = self.client.post(&url);
        let req = match &self.auth {
            AuthMode::Wallet(wallet) => req.header("X-Wallet-Auth", wallet.auth_header(path)),
            AuthMode::Bearer { token, .. } => {
                req.header("Authorization", format!("Bearer {}", token))
            }
        };

        let resp = req
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

    /// Public POST with auth, for generic JSON endpoints (DEX, etc).
    /// Takes a full URL (not a relative path) and returns the unwrapped data.
    pub async fn post_json<T: serde::de::DeserializeOwned>(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<T> {
        let req = self.client.post(url);
        let req = match &self.auth {
            AuthMode::Wallet(wallet) => {
                // Extract path from URL for auth header
                let path = url.strip_prefix(&self.base_url).unwrap_or(url);
                req.header("X-Wallet-Auth", wallet.auth_header(path))
            }
            AuthMode::Bearer { token, .. } => {
                req.header("Authorization", format!("Bearer {}", token))
            }
        };

        let resp = req
            .json(body)
            .send()
            .await
            .map_err(|e| anyhow!("Request failed: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("HTTP {}: {}", status, body_text));
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
        self.get_public("/api/v1/node/status").await
    }

    /// Fetch QUG wallet balance.
    /// v8.0.1: Uses correct endpoint with address in URL path.
    pub async fn get_balance(&self) -> Result<BalanceResponse> {
        let path = format!("/api/v1/wallets/{}/balance", self.address());
        self.get_auth(&path).await
    }

    /// Fetch all token balances. Server returns tokens as HashMap<symbol, {balance, usd_value, name, ...}>.
    pub async fn get_token_balances(&self) -> Result<Vec<crate::models::TokenBalanceDisplay>> {
        let path = "/api/v1/wallet/tokens";
        let raw: serde_json::Value = self.get_auth(path).await?;

        let mut result = Vec::new();
        if let Some(tokens_obj) = raw.get("tokens").and_then(|t| t.as_object()) {
            for (symbol, info) in tokens_obj {
                let balance = info
                    .get("balance")
                    .and_then(|b| b.as_str())
                    .unwrap_or("0")
                    .to_string();
                let usd_value = info
                    .get("usd_value")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let name = info
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or(symbol)
                    .to_string();

                // Skip tokens with zero balance
                let bal_f64: f64 = balance.parse().unwrap_or(0.0);
                if bal_f64.abs() < 1e-12 {
                    continue;
                }

                result.push(crate::models::TokenBalanceDisplay {
                    symbol: symbol.clone(),
                    name,
                    balance,
                    usd_value,
                });
            }
        }
        println!("[Tokens] Parsed {} tokens with non-zero balance", result.len());
        Ok(result)
    }

    /// Fetch all supported tokens from the public DEX endpoint (no auth needed).
    /// Returns the full token list: QUG, QUGUSD, bridge tokens (wBTC, wETH, wZEC, wIRON), + custom deployed.
    pub async fn get_supported_tokens(&self) -> Result<Vec<crate::models::SupportedToken>> {
        self.get_public::<Vec<crate::models::SupportedToken>>("/api/v1/dex/tokens").await
    }

    /// Send a transaction. When `via_mixer` is true the request is routed to
    /// `/api/v1/mixer/send` (quantum privacy mixing pool); otherwise it goes
    /// to the direct `/api/v1/transactions/send` endpoint.
    ///
    /// v8.1.7: OAuth2 Bearer users auto-sign via server vault (no mnemonic needed).
    /// v11.4.0: added `via_mixer` flag.
    pub async fn send_transaction(
        &self,
        to: &str,
        amount: &str,
        memo: Option<String>,
        mnemonic: Option<String>,
        token_type: &str,
        via_mixer: bool,
    ) -> Result<serde_json::Value> {
        let amount_f64: f64 = amount.parse().unwrap_or(0.0);

        // v8.1.7: For Bearer (OAuth2) mode, omit mnemonic to trigger vault auto-signing.
        // Server returns "vault_key_missing" error if no vault entry exists yet.
        let send_mnemonic = match &self.auth {
            AuthMode::Bearer { .. } => mnemonic, // Send mnemonic only if explicitly provided
            AuthMode::Wallet(_) => mnemonic,     // Wallet mode always sends mnemonic
        };

        // v10.2.3: Pass actual selected token type instead of hardcoding QUG.
        let body = serde_json::json!({
            "from": self.address(),
            "to": to,
            "amount": amount_f64,
            "memo": memo,
            "token_type": token_type,
            "mnemonic": send_mnemonic,
        });

        let endpoint = if via_mixer {
            "/api/v1/mixer/send"
        } else {
            "/api/v1/transactions/send"
        };
        self.post_auth(endpoint, &body).await
    }

    /// Fetch transaction history.
    /// v8.0.8: Robust parsing — logs raw response, parses records individually.
    pub async fn get_history(&self) -> Result<Vec<TransactionRecord>> {
        let path = format!("/api/v1/wallet/{}/history", self.address());
        let url = format!("{}{}", self.base_url, path);

        let req = self.client.get(&url);
        // Try with auth first (some servers require it)
        let req = match &self.auth {
            AuthMode::Wallet(wallet) => req.header("X-Wallet-Auth", wallet.auth_header(&path)),
            AuthMode::Bearer { token, .. } => {
                req.header("Authorization", format!("Bearer {}", token))
            }
        };

        let resp = req
            .send()
            .await
            .map_err(|e| anyhow!("History request failed: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("History HTTP {}: {}", status, body));
        }

        let body = resp.text().await.unwrap_or_default();
        let preview_len = body.len().min(500);
        println!(
            "[History] Raw response ({} bytes): {}",
            body.len(),
            &body[..preview_len]
        );

        // Parse as JSON value first
        let parsed: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| anyhow!("History JSON parse error: {}", e))?;

        // Find the array of transactions
        let arr = if let Some(data) = parsed.get("data") {
            if let Some(arr) = data.as_array() {
                arr
            } else if let Some(arr) = data.get("transactions").and_then(|t| t.as_array()) {
                arr
            } else {
                eprintln!("[History] Unexpected data format: {:?}", data);
                return Ok(vec![]);
            }
        } else if let Some(arr) = parsed.as_array() {
            arr
        } else {
            eprintln!("[History] No 'data' field in response");
            return Ok(vec![]);
        };

        // Parse each record individually (one bad record won't kill all)
        let mut records = Vec::new();
        for (i, item) in arr.iter().enumerate() {
            match serde_json::from_value::<TransactionRecord>(item.clone()) {
                Ok(r) => records.push(r),
                Err(e) => {
                    let raw = item.to_string();
                    eprintln!(
                        "[History] Skip record {}: {} — {}",
                        i,
                        e,
                        &raw[..raw.len().min(200)]
                    );
                }
            }
        }
        println!("[History] Parsed {}/{} records", records.len(), arr.len());
        Ok(records)
    }

    /// Fetch address book entries. Server returns {"addresses": [...], "total": N}.
    pub async fn get_address_book(&self) -> Result<Vec<crate::models::AddressBookEntry>> {
        let raw: serde_json::Value = self.get_auth("/api/v1/addressbook").await?;
        // Server wraps in {"addresses": [...], "total": N}
        let arr = raw
            .get("addresses")
            .and_then(|a| a.as_array())
            .cloned()
            .unwrap_or_default();

        let mut entries = Vec::new();
        for item in arr {
            match serde_json::from_value::<crate::models::AddressBookEntry>(item) {
                Ok(e) => entries.push(e),
                Err(e) => eprintln!("[AddressBook] Skip entry: {}", e),
            }
        }
        println!("[AddressBook] Parsed {} entries", entries.len());
        Ok(entries)
    }

    /// Save an address to the address book.
    pub async fn save_address(&self, address: &str, label: &str) -> Result<serde_json::Value> {
        let body = serde_json::json!({
            "address": address,
            "label": label,
            "favorite": false,
            "tags": [],
            "notes": "",
        });
        self.post_auth("/api/v1/addressbook", &body).await
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

    /// Submit a pool share (for PPLNS pool mining mode).
    pub async fn submit_pool_share(
        &self,
        wallet: &str,
        worker: &str,
        share_id: &str,
        difficulty: f64,
        block_height: u64,
        nonce: u64,
    ) -> Result<serde_json::Value> {
        let body = serde_json::json!({
            "wallet": wallet,
            "worker": worker,
            "share_id": share_id,
            "difficulty": difficulty,
            "block_height": block_height,
            "nonce": nonce,
            "timestamp": chrono::Utc::now().timestamp(),
        });
        let url = format!("{}/api/v1/pool/submit-share", self.base_url);
        let resp = self.client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow!("Pool share submit failed: {}", e))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Pool share HTTP {}: {}", status, text));
        }
        resp.json().await.map_err(|e| anyhow!("Pool share parse: {}", e))
    }

    /// Register this wallet as an OAuth2 client with the connected node.
    /// Only works with wallet-auth mode (requires Ed25519 signature).
    pub async fn register_oauth2_client(&self) -> Result<()> {
        if !self.has_wallet() {
            return Ok(()); // Bearer clients skip registration
        }

        let client_id = "slint-wallet";

        let registration = serde_json::json!({
            "name": "Quillon Desktop Wallet",
            "description": "Native Slint wallet OAuth2 client",
            "website": format!("http://127.0.0.1:{}", 17655),
            "redirect_uris": [
                format!("http://127.0.0.1:{}/callback", 17655),
                format!("http://localhost:{}/callback", 17655),
                "http://127.0.0.1:17655/callback",
                "http://localhost:17655/callback",
            ],
            "client_id": client_id,
            "scopes": ["read:balance", "read:history", "read:tokens", "send:transaction"],
        });

        let result: Result<serde_json::Value> =
            self.post_auth("/api/v1/oauth2/register", &registration).await;

        match result {
            Ok(_) => {
                println!("[OAuth] Registered wallet as OAuth2 client: {}", client_id);
                Ok(())
            }
            Err(e) => {
                eprintln!("[OAuth] Client registration note: {}", e);
                Ok(())
            }
        }
    }

    /// Exchange an OAuth2 authorization code for an access token (PKCE flow).
    /// This is a static method — no wallet auth needed; PKCE code_verifier proves possession.
    pub async fn exchange_oauth2_token(
        base_url: &str,
        code: &str,
        redirect_uri: &str,
        code_verifier: &str,
    ) -> Result<crate::models::OAuthTokenResponse> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .map_err(|e| anyhow!("HTTP client error: {}", e))?;

        let url = format!("{}/api/v1/oauth2/token", base_url.trim_end_matches('/'));

        let body = serde_json::json!({
            "grant_type": "authorization_code",
            "client_id": "slint-wallet",
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        });

        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow!("Token exchange request failed: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Token exchange HTTP {}: {}", status, text));
        }

        let wrapper: ApiWrapper<crate::models::OAuthTokenResponse> = resp
            .json()
            .await
            .map_err(|e| anyhow!("Token response parse error: {}", e))?;

        if let Some(err) = wrapper.error.filter(|e| !e.is_empty()) {
            return Err(anyhow!("Token exchange error: {}", err));
        }

        wrapper.data.ok_or_else(|| anyhow!("No data in token response"))
    }

    /// Fetch user info using a Bearer access token (no wallet auth).
    pub async fn get_userinfo_with_token(
        base_url: &str,
        access_token: &str,
    ) -> Result<crate::models::OAuthUserInfo> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .map_err(|e| anyhow!("HTTP client error: {}", e))?;

        let url = format!("{}/api/v1/oauth2/userinfo", base_url.trim_end_matches('/'));

        let resp = client
            .get(&url)
            .header("Authorization", format!("Bearer {}", access_token))
            .send()
            .await
            .map_err(|e| anyhow!("Userinfo request failed: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Userinfo HTTP {}: {}", status, text));
        }

        let wrapper: ApiWrapper<crate::models::OAuthUserInfo> = resp
            .json()
            .await
            .map_err(|e| anyhow!("Userinfo parse error: {}", e))?;

        if let Some(err) = wrapper.error.filter(|e| !e.is_empty()) {
            return Err(anyhow!("Userinfo error: {}", err));
        }

        wrapper.data.ok_or_else(|| anyhow!("No data in userinfo response"))
    }

    /// Submit OAuth2 consent to the backend and receive an authorization code.
    /// Only works with wallet-auth mode (requires Ed25519 signature).
    pub async fn authorize_oauth2(
        &self,
        client_id: &str,
        scopes: &[String],
        redirect_uri: &str,
        code_challenge: Option<&str>,
        code_challenge_method: Option<&str>,
    ) -> Result<String> {
        let consent = serde_json::json!({
            "wallet_address": self.address(),
            "client_id": client_id,
            "scopes": scopes,
            "approved": true,
            "auth_request_id": format!("slint-{}", chrono::Utc::now().timestamp_millis()),
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
        });

        let resp: serde_json::Value =
            self.post_auth("/api/v1/oauth2/consent", &consent).await?;

        resp.get("auth_code")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("No auth_code in consent response"))
    }
}
