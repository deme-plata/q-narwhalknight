/// Multi-Bootstrap Configuration for Q-NarwhalKnight
/// v2.9.0-beta: Decentralized bootstrap with automatic failover
///
/// Supports multiple bootstrap servers for true decentralization:
/// - Automatic failover when primary is down
/// - Parallel health checking
/// - Configurable via environment variables
///
/// Environment Variables:
/// - Q_BOOTSTRAP_URLS: Comma-separated list of bootstrap URLs
///   Example: "http://185.182.185.227:8080,http://161.35.219.10:8080"
/// - Q_BOOTSTRAP_URL: Legacy single-URL support (still works)
/// - Q_BOOTSTRAP_TIMEOUT_MS: Timeout for each server (default: 5000)

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Default bootstrap servers — Delta 1Gbit first for fastest sync
pub const DEFAULT_BOOTSTRAP_SERVERS: &[&str] = &[
    "http://5.79.79.158:8080",     // Server Delta (primary - 1Gbit fastest)
    "http://109.205.176.60:8080",  // Server Gamma (secondary - 1Gbit)
    "http://185.182.185.227:8080", // Server Beta (tertiary - 100Mbit)
    "http://161.35.219.10:8082",   // Server Alpha (quaternary/testing) - port 8082
];

/// Bootstrap server with health status
#[derive(Debug, Clone)]
pub struct BootstrapServer {
    pub url: String,
    pub is_healthy: bool,
    pub last_check: std::time::Instant,
    pub response_time_ms: u64,
    pub height: u64,
}

impl BootstrapServer {
    pub fn new(url: String) -> Self {
        Self {
            url,
            is_healthy: false,
            last_check: std::time::Instant::now(),
            response_time_ms: 0,
            height: 0,
        }
    }
}

/// Multi-bootstrap configuration with automatic failover
pub struct BootstrapConfig {
    servers: Arc<RwLock<Vec<BootstrapServer>>>,
    timeout: Duration,
    client: reqwest::Client,
}

impl BootstrapConfig {
    /// Create new bootstrap config from environment or defaults
    pub fn from_env() -> Self {
        let timeout_ms = std::env::var("Q_BOOTSTRAP_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(5000);

        let timeout = Duration::from_millis(timeout_ms);

        // Try Q_BOOTSTRAP_URLS first (comma-separated list)
        let urls: Vec<String> = if let Ok(urls_str) = std::env::var("Q_BOOTSTRAP_URLS") {
            urls_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        } else if let Ok(single_url) = std::env::var("Q_BOOTSTRAP_URL") {
            // Legacy single-URL support
            vec![single_url]
        } else {
            // Use defaults
            DEFAULT_BOOTSTRAP_SERVERS
                .iter()
                .map(|s| s.to_string())
                .collect()
        };

        let servers: Vec<BootstrapServer> = urls
            .into_iter()
            .map(BootstrapServer::new)
            .collect();

        info!("🌐 Bootstrap configuration initialized with {} servers:", servers.len());
        for (i, server) in servers.iter().enumerate() {
            info!("   {}. {}", i + 1, server.url);
        }

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(4)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            servers: Arc::new(RwLock::new(servers)),
            timeout,
            client,
        }
    }

    /// Get the best available bootstrap server (with failover)
    pub async fn get_best_server(&self) -> Option<String> {
        let servers = self.servers.read().await;

        // First, try to find a healthy server with the highest height
        let healthy_servers: Vec<_> = servers
            .iter()
            .filter(|s| s.is_healthy)
            .collect();

        if let Some(best) = healthy_servers.iter().max_by_key(|s| s.height) {
            return Some(best.url.clone());
        }

        // If no healthy servers, return the first one (will trigger health check)
        servers.first().map(|s| s.url.clone())
    }

    /// Get all bootstrap URLs for parallel operations
    pub async fn get_all_urls(&self) -> Vec<String> {
        let servers = self.servers.read().await;
        servers.iter().map(|s| s.url.clone()).collect()
    }

    /// Get healthy servers only
    pub async fn get_healthy_servers(&self) -> Vec<String> {
        let servers = self.servers.read().await;
        servers
            .iter()
            .filter(|s| s.is_healthy)
            .map(|s| s.url.clone())
            .collect()
    }

    /// Check health of all bootstrap servers
    pub async fn health_check_all(&self) -> usize {
        let urls = self.get_all_urls().await;
        let mut healthy_count = 0;

        for url in urls {
            if self.check_server_health(&url).await {
                healthy_count += 1;
            }
        }

        info!("🏥 Bootstrap health check: {}/{} servers healthy",
              healthy_count, self.servers.read().await.len());

        healthy_count
    }

    /// Check health of a single server
    pub async fn check_server_health(&self, url: &str) -> bool {
        let start = std::time::Instant::now();
        let status_url = format!("{}/api/v1/status", url);

        match self.client.get(&status_url).send().await {
            Ok(resp) if resp.status().is_success() => {
                let response_time = start.elapsed().as_millis() as u64;

                // Try to parse the response to get height
                let height = if let Ok(json) = resp.json::<serde_json::Value>().await {
                    json.get("data")
                        .and_then(|d| d.get("height"))
                        .and_then(|h| h.as_u64())
                        .unwrap_or(0)
                } else {
                    0
                };

                // Update server status
                let mut servers = self.servers.write().await;
                if let Some(server) = servers.iter_mut().find(|s| s.url == url) {
                    server.is_healthy = true;
                    server.last_check = std::time::Instant::now();
                    server.response_time_ms = response_time;
                    server.height = height;
                }

                debug!("✅ {} healthy ({}ms, height: {})", url, response_time, height);
                true
            }
            Ok(resp) => {
                warn!("⚠️ {} returned error status: {}", url, resp.status());
                self.mark_unhealthy(url).await;
                false
            }
            Err(e) => {
                warn!("❌ {} unreachable: {}", url, e);
                self.mark_unhealthy(url).await;
                false
            }
        }
    }

    /// Mark a server as unhealthy
    async fn mark_unhealthy(&self, url: &str) {
        let mut servers = self.servers.write().await;
        if let Some(server) = servers.iter_mut().find(|s| s.url == url) {
            server.is_healthy = false;
            server.last_check = std::time::Instant::now();
        }
    }

    /// Try an operation on bootstrap servers with automatic failover
    /// Returns the result from the first successful server
    pub async fn try_with_failover<T, F, Fut>(&self, operation: F) -> Option<(String, T)>
    where
        F: Fn(String) -> Fut,
        Fut: std::future::Future<Output = Option<T>>,
    {
        // Clone URLs to release read lock before async operations
        let sorted_urls: Vec<String> = {
            let servers = self.servers.read().await;
            let mut sorted: Vec<_> = servers.iter().collect();
            sorted.sort_by(|a, b| {
                match (a.is_healthy, b.is_healthy) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => a.response_time_ms.cmp(&b.response_time_ms),
                }
            });
            sorted.iter().map(|s| s.url.clone()).collect()
        };

        for url in sorted_urls {
            let url = url.clone();
            debug!("🔄 Trying bootstrap server: {}", url);

            if let Some(result) = operation(url.clone()).await {
                // Mark as healthy on success
                let mut servers = self.servers.write().await;
                if let Some(s) = servers.iter_mut().find(|s| s.url == url) {
                    s.is_healthy = true;
                    s.last_check = std::time::Instant::now();
                }
                return Some((url, result));
            } else {
                // Mark as potentially unhealthy
                warn!("⚠️ Bootstrap server {} failed, trying next...", url);
            }
        }

        error!("❌ All bootstrap servers failed!");
        None
    }

    /// Fetch genesis block with failover
    pub async fn fetch_genesis_with_failover(&self) -> Option<q_types::Block> {
        self.try_with_failover(|url| async move {
            let genesis_url = format!("{}/api/v1/blocks/1", url);

            match reqwest::get(&genesis_url).await {
                Ok(resp) if resp.status().is_success() => {
                    match resp.json::<crate::handlers::ApiResponse<q_types::Block>>().await {
                        Ok(api_resp) => api_resp.data,
                        Err(e) => {
                            warn!("Failed to parse genesis from {}: {}", url, e);
                            None
                        }
                    }
                }
                _ => None,
            }
        })
        .await
        .map(|(url, block)| {
            info!("✅ Genesis block fetched from {}", url);
            block
        })
    }

    /// Fetch blocks with failover
    pub async fn fetch_blocks_with_failover(
        &self,
        from_height: u64,
        limit: u64,
    ) -> Option<Vec<q_types::Block>> {
        self.try_with_failover(|url| async move {
            let blocks_url = format!(
                "{}/api/v1/sync/blocks?from_height={}&limit={}",
                url, from_height, limit
            );

            match reqwest::get(&blocks_url).await {
                Ok(resp) if resp.status().is_success() => {
                    match resp.json::<crate::handlers::ApiResponse<Vec<q_types::Block>>>().await {
                        Ok(api_resp) => api_resp.data,
                        Err(_) => None,
                    }
                }
                _ => None,
            }
        })
        .await
        .map(|(url, blocks)| {
            debug!("📦 Fetched {} blocks from {}", blocks.len(), url);
            blocks
        })
    }

    /// Get network height with failover (uses highest reported height)
    pub async fn get_network_height_with_failover(&self) -> Option<u64> {
        // Check all servers and use the highest height
        self.health_check_all().await;

        let servers = self.servers.read().await;
        servers
            .iter()
            .filter(|s| s.is_healthy)
            .map(|s| s.height)
            .max()
    }

    /// Get status for monitoring
    pub async fn get_status(&self) -> Vec<serde_json::Value> {
        let servers = self.servers.read().await;
        servers
            .iter()
            .map(|s| {
                serde_json::json!({
                    "url": s.url,
                    "healthy": s.is_healthy,
                    "response_time_ms": s.response_time_ms,
                    "height": s.height,
                    "last_check_ago_ms": s.last_check.elapsed().as_millis()
                })
            })
            .collect()
    }
}

/// Global bootstrap config instance
static BOOTSTRAP_CONFIG: std::sync::OnceLock<BootstrapConfig> = std::sync::OnceLock::new();

/// Get or initialize the global bootstrap config
pub fn get_bootstrap_config() -> &'static BootstrapConfig {
    BOOTSTRAP_CONFIG.get_or_init(BootstrapConfig::from_env)
}

/// Helper function: Get the best bootstrap URL (with automatic failover)
pub async fn get_bootstrap_url() -> String {
    get_bootstrap_config()
        .get_best_server()
        .await
        .unwrap_or_else(|| DEFAULT_BOOTSTRAP_SERVERS[0].to_string())
}

/// Helper function: Get all bootstrap URLs
pub async fn get_all_bootstrap_urls() -> Vec<String> {
    get_bootstrap_config().get_all_urls().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_servers() {
        assert!(DEFAULT_BOOTSTRAP_SERVERS.len() >= 2);
        assert!(DEFAULT_BOOTSTRAP_SERVERS[0].contains("185.182.185.227"));
        assert!(DEFAULT_BOOTSTRAP_SERVERS[1].contains("161.35.219.10"));
    }

    #[test]
    fn test_bootstrap_server_new() {
        let server = BootstrapServer::new("http://example.com:8080".to_string());
        assert_eq!(server.url, "http://example.com:8080");
        assert!(!server.is_healthy);
        assert_eq!(server.height, 0);
    }

    #[tokio::test]
    async fn test_config_from_env() {
        // Uses defaults when no env vars set
        let config = BootstrapConfig::from_env();
        let urls = config.get_all_urls().await;
        assert!(!urls.is_empty());
    }
}
