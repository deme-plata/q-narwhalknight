/// HTTP client for Quillon Bank API

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::config::CliConfig;

pub struct QuilonBankClient {
    client: Client,
    base_url: String,
    auth_token: Option<String>,
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

        Ok(Self {
            client,
            base_url: config.node.api_endpoint.clone(),
            auth_token: None,
        })
    }

    pub fn set_auth_token(&mut self, token: String) {
        self.auth_token = Some(token);
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