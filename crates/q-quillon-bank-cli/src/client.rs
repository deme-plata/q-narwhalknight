/// HTTP client for Quillon Bank API with AEGIS-QL authentication

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use q_aegis_ql::Signature as AegisSignature;

use crate::config::CliConfig;
use crate::auth::AuthManager;

pub struct QuilonBankClient {
    client: Client,
    base_url: String,
    auth_token: Option<String>,
    auth_manager: Option<AuthManager>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl QuilonBankClient {
    pub fn new(config: &CliConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.node.timeout))
            .build()
            .context("Failed to create HTTP client")?;

        // Initialize with AuthManager for AEGIS-QL signing
        let auth_manager = AuthManager::new(config.clone());

        Ok(Self {
            client,
            base_url: config.node.api_endpoint.clone(),
            auth_token: None,
            auth_manager: Some(auth_manager),
        })
    }

    pub fn set_auth_token(&mut self, token: String) {
        self.auth_token = Some(token);
    }

    /// Sign an operation and get AEGIS-QL signature with wallet address
    fn sign_operation(&self, operation: &str) -> Result<(AegisSignature, [u8; 32], i64)> {
        let auth_manager = self.auth_manager.as_ref()
            .context("Auth manager not initialized")?;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs() as i64;

        let (signature, wallet) = auth_manager.sign_operation_aegis(operation, timestamp)?;

        Ok((signature, wallet, timestamp))
    }

    pub async fn get<T: for<'de> Deserialize<'de>>(&self, endpoint: &str) -> Result<ApiResponse<T>> {
        let url = format!("{}{}", self.base_url, endpoint);

        let mut request = self.client.get(&url);

        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }

        let response = request
            .send()
            .await
            .context("Failed to send request")?;

        response
            .json::<ApiResponse<T>>()
            .await
            .context("Failed to parse response")
    }

    pub async fn post<T: for<'de> Deserialize<'de>, B: Serialize>(
        &self,
        endpoint: &str,
        body: &B,
    ) -> Result<ApiResponse<T>> {
        let url = format!("{}{}", self.base_url, endpoint);

        let mut request = self.client.post(&url).json(body);

        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }

        let response = request
            .send()
            .await
            .context("Failed to send request")?;

        response
            .json::<ApiResponse<T>>()
            .await
            .context("Failed to parse response")
    }

    /// POST request with AEGIS-QL authentication
    pub async fn post_with_auth<T: for<'de> Deserialize<'de>, B: Serialize>(
        &self,
        endpoint: &str,
        body: &B,
        operation: &str,
    ) -> Result<ApiResponse<T>> {
        // Sign the operation with AEGIS-QL
        let (signature, wallet, timestamp) = self.sign_operation(operation)?;

        let url = format!("{}{}", self.base_url, endpoint);

        // Serialize signature to bytes
        let signature_bytes = signature.to_bytes();

        // Build request with AEGIS-QL headers
        let request = self.client.post(&url)
            .json(body)
            .header("X-Wallet-Address", hex::encode(wallet))
            .header("X-AEGIS-Signature", hex::encode(&signature_bytes))
            .header("X-Timestamp", timestamp.to_string())
            .header("X-Operation", operation)
            .header("Content-Type", "application/json");

        let response = request
            .send()
            .await
            .context("Failed to send authenticated request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }

        response
            .json::<ApiResponse<T>>()
            .await
            .context("Failed to parse response")
    }

    /// GET request with AEGIS-QL authentication
    pub async fn get_with_auth<T: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        operation: &str,
    ) -> Result<ApiResponse<T>> {
        // Sign the operation with AEGIS-QL
        let (signature, wallet, timestamp) = self.sign_operation(operation)?;

        let url = format!("{}{}", self.base_url, endpoint);

        // Serialize signature to bytes
        let signature_bytes = signature.to_bytes();

        // Build request with AEGIS-QL headers
        let request = self.client.get(&url)
            .header("X-Wallet-Address", hex::encode(wallet))
            .header("X-AEGIS-Signature", hex::encode(&signature_bytes))
            .header("X-Timestamp", timestamp.to_string())
            .header("X-Operation", operation);

        let response = request
            .send()
            .await
            .context("Failed to send authenticated request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }

        response
            .json::<ApiResponse<T>>()
            .await
            .context("Failed to parse response")
    }

    pub async fn put<T: for<'de> Deserialize<'de>, B: Serialize>(
        &self,
        endpoint: &str,
        body: &B,
    ) -> Result<ApiResponse<T>> {
        let url = format!("{}{}", self.base_url, endpoint);

        let mut request = self.client.put(&url).json(body);

        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }

        let response = request
            .send()
            .await
            .context("Failed to send request")?;

        response
            .json::<ApiResponse<T>>()
            .await
            .context("Failed to parse response")
    }

    pub async fn delete<T: for<'de> Deserialize<'de>>(&self, endpoint: &str) -> Result<ApiResponse<T>> {
        let url = format!("{}{}", self.base_url, endpoint);

        let mut request = self.client.delete(&url);

        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }

        let response = request
            .send()
            .await
            .context("Failed to send request")?;

        response
            .json::<ApiResponse<T>>()
            .await
            .context("Failed to parse response")
    }
}