//! PaaS API client implementation

use crate::{error::{PaaSError, Result}, types::*};
use reqwest::{Client, header};
use serde_json::json;
use uuid::Uuid;

/// Q-NarwhalKnight PaaS API Client
pub struct PaaSClient {
    api_key: String,
    base_url: String,
    client: Client,
}

impl PaaSClient {
    /// Create a new PaaS client
    pub fn new(api_key: String, base_url: String) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            api_key,
            base_url,
            client,
        }
    }

    /// Mix a Bitcoin transaction
    pub async fn mix_bitcoin_transaction(
        &self,
        signed_tx_hex: &str,
        privacy_level: PrivacyLevel,
        use_tor: bool,
    ) -> Result<MixResult> {
        let url = format!("{}/api/v1/privacy/mix/submit", self.base_url);
        
        let payload = json!({
            "chain": "bitcoin",
            "transaction_hex": signed_tx_hex,
            "privacy_level": privacy_level.to_string(),
            "use_tor": use_tor,
        });

        self.post(&url, payload).await
    }

    /// Mix an Ethereum transaction
    pub async fn mix_ethereum_transaction(
        &self,
        signed_tx_hex: &str,
        privacy_level: PrivacyLevel,
        mev_protection: bool,
    ) -> Result<MixResult> {
        let url = format!("{}/api/v1/privacy/mix/submit", self.base_url);
        
        let payload = json!({
            "chain": "ethereum",
            "transaction_hex": signed_tx_hex,
            "privacy_level": privacy_level.to_string(),
            "mev_protection": mev_protection,
        });

        self.post(&url, payload).await
    }

    /// Get mix status
    pub async fn get_mix_status(&self, mix_id: &str) -> Result<MixResult> {
        let url = format!("{}/api/v1/privacy/mix/status/{}", self.base_url, mix_id);
        self.get(&url).await
    }

    /// Get billing information
    pub async fn get_billing_info(&self) -> Result<BillingInfo> {
        let url = format!("{}/api/v1/billing/balance", self.base_url);
        self.get(&url).await
    }

    /// Get API key information
    pub async fn get_api_key_info(&self) -> Result<ApiKeyInfo> {
        let url = format!("{}/api/v1/privacy/paas/api-keys/info", self.base_url);
        self.get(&url).await
    }

    // Internal helper methods
    async fn post<T: serde::de::DeserializeOwned>(
        &self,
        url: &str,
        payload: serde_json::Value,
    ) -> Result<T> {
        let idempotency_key = Uuid::new_v4().to_string();

        let response = self.client
            .post(url)
            .header(header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(header::CONTENT_TYPE, "application/json")
            .header("Idempotency-Key", idempotency_key)
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(PaaSError::ApiError(format!("HTTP {}: {}", status, error_text)));
        }

        let api_response: ApiResponse<T> = response.json().await?;

        if !api_response.success {
            return Err(PaaSError::ApiError(
                api_response.error.unwrap_or_else(|| "Unknown error".to_string())
            ));
        }

        api_response.data.ok_or_else(|| PaaSError::ApiError("No data in response".to_string()))
    }

    async fn get<T: serde::de::DeserializeOwned>(&self, url: &str) -> Result<T> {
        let response = self.client
            .get(url)
            .header(header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(header::CONTENT_TYPE, "application/json")
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(PaaSError::ApiError(format!("HTTP {}: {}", status, error_text)));
        }

        let api_response: ApiResponse<T> = response.json().await?;

        if !api_response.success {
            return Err(PaaSError::ApiError(
                api_response.error.unwrap_or_else(|| "Unknown error".to_string())
            ));
        }

        api_response.data.ok_or_else(|| PaaSError::ApiError("No data in response".to_string()))
    }
}
