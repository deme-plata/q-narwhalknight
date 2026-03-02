/// HTTP client for Dune Analytics API.
/// Handles table creation and CSV data insertion with rate limiting and retries.

use anyhow::{anyhow, Result};
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::schema::TableDef;

const DUNE_BASE_URL: &str = "https://api.dune.com/api/v1";
const MAX_RETRIES: u32 = 5;
const RATE_LIMIT_DELAY: Duration = Duration::from_millis(2500); // Dune free plan: ~20 req/min

#[derive(Clone)]
pub struct DuneClient {
    http: reqwest::Client,
    api_key: String,
    namespace: String,
}

impl DuneClient {
    pub fn new(api_key: String, namespace: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client");
        Self { http, api_key, namespace }
    }

    /// Create a custom table on Dune if it doesn't exist.
    pub async fn create_table(&self, table: &TableDef) -> Result<()> {
        let url = format!("{}/table/create", DUNE_BASE_URL);

        let schema: Vec<serde_json::Value> = table.columns.iter().map(|c| {
            serde_json::json!({
                "name": c.name,
                "type": c.dune_type,
            })
        }).collect();

        let body = serde_json::json!({
            "namespace": self.namespace,
            "table_name": table.name,
            "description": table.description,
            "schema": schema,
            "is_private": false,
        });

        let resp = self.http.post(&url)
            .header("X-DUNE-API-KEY", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if status.is_success() {
            info!("[Dune] Created table {}.{}", self.namespace, table.name);
            Ok(())
        } else if status.as_u16() == 409 {
            debug!("[Dune] Table {}.{} already exists", self.namespace, table.name);
            Ok(())
        } else {
            let text = resp.text().await.unwrap_or_default();
            // Some Dune API versions return 200 with error, or 400 if table exists
            if text.contains("already exists") {
                debug!("[Dune] Table {}.{} already exists", self.namespace, table.name);
                Ok(())
            } else {
                Err(anyhow!("[Dune] create_table {} failed ({}): {}", table.name, status, text))
            }
        }
    }

    /// Insert CSV data into a Dune custom table.
    /// Returns the number of rows inserted on success.
    pub async fn insert_csv(&self, table_name: &str, csv_data: &str) -> Result<u64> {
        if csv_data.is_empty() {
            return Ok(0);
        }

        let url = format!("{}/table/{}/{}/insert", DUNE_BASE_URL, self.namespace, table_name);
        let mut retries = 0u32;
        let mut delay = Duration::from_secs(5); // Initial retry delay for 429s

        loop {
            let resp = self.http.post(&url)
                .header("X-DUNE-API-KEY", &self.api_key)
                .header("Content-Type", "text/csv")
                .body(csv_data.to_owned())
                .send()
                .await;

            match resp {
                Ok(r) => {
                    let status = r.status();
                    if status.is_success() {
                        let body: serde_json::Value = r.json().await.unwrap_or_default();
                        let rows = body.get("rows_written")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        debug!("[Dune] Inserted {} rows into {}", rows, table_name);
                        // Rate limit: wait between requests
                        tokio::time::sleep(RATE_LIMIT_DELAY).await;
                        return Ok(rows);
                    }

                    let status_code = status.as_u16();
                    let text = r.text().await.unwrap_or_default();

                    if (status_code == 429 || status_code >= 500) && retries < MAX_RETRIES {
                        retries += 1;
                        warn!("[Dune] {} inserting into {} (attempt {}/{}), retrying in {:?}: {}",
                              status_code, table_name, retries, MAX_RETRIES, delay, text);
                        tokio::time::sleep(delay).await;
                        delay = delay.mul_f32(2.0).min(Duration::from_secs(60));
                        continue;
                    }

                    return Err(anyhow!("[Dune] insert_csv {} failed ({}): {}", table_name, status_code, text));
                }
                Err(e) => {
                    if retries < MAX_RETRIES {
                        retries += 1;
                        warn!("[Dune] Network error inserting into {} (attempt {}/{}): {}",
                              table_name, retries, MAX_RETRIES, e);
                        tokio::time::sleep(delay).await;
                        delay = delay.mul_f32(2.0).min(Duration::from_secs(60));
                        continue;
                    }
                    return Err(anyhow!("[Dune] insert_csv {} network error: {}", table_name, e));
                }
            }
        }
    }
}
