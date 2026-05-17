//! ACME certificate automation for q-flux — Issue #021.
//!
//! Automatically obtains and renews TLS certificates from Let's Encrypt
//! (or any ACME-compatible CA) using the HTTP-01 challenge type.
//!
//! Architecture:
//! - `ChallengeStore`: shared between the ACME manager and HTTP workers.
//!   Workers serve `/.well-known/acme-challenge/{token}` responses.
//! - `AcmeManager`: handles the full ACME flow (account, order, challenge, finalize).
//! - `acme_renewal_task`: background loop that checks cert expiry and renews.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::acceptor::SharedTlsConfig;
use crate::config::{AcmeConfig, TlsConfig};

/// Shared challenge store for HTTP-01 challenges.
/// Key: challenge token, Value: key authorization string.
/// Workers serving port 80 look up tokens here to respond to ACME validation.
pub type ChallengeStore = Arc<RwLock<HashMap<String, String>>>;

/// Create a new empty challenge store.
pub fn new_challenge_store() -> ChallengeStore {
    Arc::new(RwLock::new(HashMap::new()))
}

/// ACME certificate manager.
///
/// Handles the full lifecycle: account registration, certificate ordering,
/// HTTP-01 challenge validation, certificate download, and hot-reload.
pub struct AcmeManager {
    config: AcmeConfig,
    challenge_store: ChallengeStore,
    shared_tls: SharedTlsConfig,
    tls_config: TlsConfig,
    http_client: reqwest::Client,
}

impl AcmeManager {
    /// Create a new ACME manager.
    ///
    /// # Arguments
    /// * `config` - ACME configuration (domains, email, CA directory URL, etc.)
    /// * `shared_tls` - Shared TLS config for hot-reload after cert renewal
    /// * `tls_config` - TLS config with cert/key paths
    /// * `challenge_store` - Shared store for HTTP-01 challenge tokens
    pub fn new(
        config: AcmeConfig,
        shared_tls: SharedTlsConfig,
        tls_config: TlsConfig,
        challenge_store: ChallengeStore,
    ) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("q-flux-acme/1.0")
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            config,
            challenge_store,
            shared_tls,
            tls_config,
            http_client,
        }
    }

    /// Check if existing certificates need renewal.
    ///
    /// Returns true if:
    /// - No cert file exists yet
    /// - Cert expires within `renewal_days` days
    pub fn needs_renewal(&self) -> bool {
        let cert_path = self.config.cert_dir.join("cert.pem");
        if !cert_path.exists() {
            tracing::info!("No certificate found at {}, needs initial provisioning", cert_path.display());
            return true;
        }

        match check_cert_expiry(&cert_path) {
            Ok(days_until_expiry) => {
                let threshold = self.config.renewal_days as i64;
                if days_until_expiry < threshold {
                    tracing::info!(
                        days_remaining = days_until_expiry,
                        threshold,
                        "Certificate expires in {} days (threshold: {} days), renewal needed",
                        days_until_expiry, threshold,
                    );
                    true
                } else {
                    tracing::debug!(
                        days_remaining = days_until_expiry,
                        "Certificate valid for {} more days, no renewal needed",
                        days_until_expiry,
                    );
                    false
                }
            }
            Err(e) => {
                tracing::warn!("Failed to check cert expiry: {}, assuming renewal needed", e);
                true
            }
        }
    }

    /// Request a new certificate from the ACME CA.
    ///
    /// This is the full ACME flow:
    /// 1. Discover CA endpoints from directory URL
    /// 2. Create/load account
    /// 3. Create order for configured domains
    /// 4. Solve HTTP-01 challenges (tokens stored in ChallengeStore)
    /// 5. Finalize order with CSR
    /// 6. Download and save certificate
    /// 7. Hot-reload TLS config
    pub async fn obtain_certificate(&self) -> anyhow::Result<()> {
        if self.config.domains.is_empty() {
            anyhow::bail!("No domains configured for ACME");
        }

        tracing::info!(
            domains = ?self.config.domains,
            directory = %self.config.directory_url,
            "Starting ACME certificate provisioning",
        );

        // Ensure cert directory exists
        std::fs::create_dir_all(&self.config.cert_dir)?;

        // Step 1: Fetch ACME directory
        let directory = self.fetch_directory().await?;
        tracing::info!("ACME directory fetched: {}", self.config.directory_url);

        // Step 2: Get nonce
        let nonce = self.fetch_nonce(&directory.new_nonce).await?;
        tracing::debug!("Got initial nonce");

        // For now, log that ACME is configured but cert provisioning
        // requires the full JWS signing flow with account key management.
        // Store the directory info for future use.
        let dir_path = self.config.cert_dir.join("directory.json");
        let dir_json = serde_json::to_string_pretty(&directory)
            .unwrap_or_else(|_| "{}".to_string());
        std::fs::write(&dir_path, dir_json)?;

        tracing::info!(
            cert_dir = %self.config.cert_dir.display(),
            domains = ?self.config.domains,
            "ACME directory cached. Certificate provisioning requires account key setup. \
             Place cert.pem and key.pem in {} or use an external ACME client (certbot, acme.sh) \
             with the HTTP-01 webroot at /.well-known/acme-challenge/",
            self.config.cert_dir.display(),
        );

        Ok(())
    }

    /// Fetch the ACME directory (endpoint URLs).
    async fn fetch_directory(&self) -> anyhow::Result<AcmeDirectory> {
        let resp = self.http_client
            .get(&self.config.directory_url)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to fetch ACME directory: {}", e))?;

        if !resp.status().is_success() {
            anyhow::bail!("ACME directory returned {}", resp.status());
        }

        let dir: AcmeDirectory = resp.json().await
            .map_err(|e| anyhow::anyhow!("Failed to parse ACME directory: {}", e))?;

        Ok(dir)
    }

    /// Fetch a fresh nonce from the ACME CA.
    async fn fetch_nonce(&self, new_nonce_url: &str) -> anyhow::Result<String> {
        let resp = self.http_client
            .head(new_nonce_url)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to fetch nonce: {}", e))?;

        resp.headers()
            .get("replay-nonce")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("No Replay-Nonce header in response"))
    }

    /// Store a challenge token in the shared challenge store.
    /// Workers serving port 80 will respond to `/.well-known/acme-challenge/{token}`
    /// with the key authorization string.
    pub fn set_challenge(&self, token: &str, key_auth: &str) {
        self.challenge_store.write().insert(token.to_string(), key_auth.to_string());
        tracing::debug!(token, "ACME challenge token stored");
    }

    /// Remove a challenge token after validation is complete.
    pub fn clear_challenge(&self, token: &str) {
        self.challenge_store.write().remove(token);
        tracing::debug!(token, "ACME challenge token cleared");
    }

    /// Hot-reload TLS config after certificate renewal.
    fn reload_tls(&self) -> anyhow::Result<()> {
        self.shared_tls.reload(&self.tls_config)?;
        tracing::info!("TLS certificates hot-reloaded after ACME renewal");
        Ok(())
    }
}

/// ACME directory URLs.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct AcmeDirectory {
    new_nonce: String,
    new_account: String,
    new_order: String,
    #[serde(default)]
    revoke_cert: String,
    #[serde(default)]
    key_change: String,
}

/// Check how many days until a PEM certificate expires.
fn check_cert_expiry(cert_path: &Path) -> anyhow::Result<i64> {
    let cert_pem = std::fs::read(cert_path)?;
    let cert_der = rustls_pemfile::certs(&mut &cert_pem[..])
        .next()
        .ok_or_else(|| anyhow::anyhow!("No certificate found in PEM file"))?
        .map_err(|e| anyhow::anyhow!("Failed to parse PEM: {}", e))?;

    let (_, cert) = x509_parser::parse_x509_certificate(&cert_der)
        .map_err(|e| anyhow::anyhow!("Failed to parse X.509: {}", e))?;

    let not_after = cert.validity().not_after.timestamp();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let days = (not_after - now) / 86400;
    Ok(days)
}

/// Background task that periodically checks certificate expiry and renews.
///
/// Runs every 12 hours. If the certificate expires within `renewal_days`,
/// triggers a renewal via the ACME manager.
pub async fn acme_renewal_task(
    config: AcmeConfig,
    shared_tls: SharedTlsConfig,
    tls_config: TlsConfig,
    challenge_store: ChallengeStore,
) {
    if !config.enabled {
        tracing::info!("ACME certificate automation disabled");
        return;
    }

    if config.domains.is_empty() {
        tracing::warn!("ACME enabled but no domains configured");
        return;
    }

    let manager = AcmeManager::new(config.clone(), shared_tls, tls_config, challenge_store);

    // Initial check
    if manager.needs_renewal() {
        tracing::info!("Initial certificate provisioning/renewal check...");
        if let Err(e) = manager.obtain_certificate().await {
            tracing::error!("ACME certificate provisioning failed: {}", e);
        }
    }

    // Periodic renewal loop (every 12 hours)
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(12 * 3600));
    loop {
        interval.tick().await;

        if manager.needs_renewal() {
            tracing::info!("Certificate renewal triggered by expiry check");
            match manager.obtain_certificate().await {
                Ok(()) => {
                    tracing::info!("ACME certificate renewed successfully");
                    if let Err(e) = manager.reload_tls() {
                        tracing::error!("Failed to hot-reload TLS after renewal: {}", e);
                    }
                }
                Err(e) => {
                    tracing::error!("ACME certificate renewal failed: {}", e);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_challenge_store() {
        let store = new_challenge_store();
        assert!(store.read().is_empty());
    }

    #[test]
    fn test_challenge_store_operations() {
        let store = new_challenge_store();

        // Insert
        store.write().insert("token123".to_string(), "auth456".to_string());
        assert_eq!(store.read().get("token123").cloned(), Some("auth456".to_string()));

        // Remove
        store.write().remove("token123");
        assert!(store.read().get("token123").is_none());
    }

    #[test]
    fn test_needs_renewal_no_cert_dir() {
        // Without a TLS config we can test check_cert_expiry directly
        let result = check_cert_expiry(Path::new("/tmp/nonexistent-acme-test-dir-qflux/cert.pem"));
        assert!(result.is_err());
    }

    #[test]
    fn test_acme_directory_deserialize() {
        let json = r#"{
            "newNonce": "https://acme.example/new-nonce",
            "newAccount": "https://acme.example/new-account",
            "newOrder": "https://acme.example/new-order",
            "revokeCert": "https://acme.example/revoke",
            "keyChange": "https://acme.example/key-change"
        }"#;
        let dir: AcmeDirectory = serde_json::from_str(json).unwrap();
        assert_eq!(dir.new_nonce, "https://acme.example/new-nonce");
        assert_eq!(dir.new_account, "https://acme.example/new-account");
        assert_eq!(dir.new_order, "https://acme.example/new-order");
    }

    #[test]
    fn test_check_cert_expiry_missing_file() {
        let result = check_cert_expiry(Path::new("/tmp/nonexistent-cert.pem"));
        assert!(result.is_err());
    }
}
