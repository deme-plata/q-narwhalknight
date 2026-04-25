use q_types::NodeId;
use serde::{Deserialize, Serialize};
use std::env;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub port: u16,
    pub host: String,
    pub is_validator: bool,
    pub p2p_port: u16,
    pub bootstrap_peers: Vec<String>,
    pub database_url: Option<String>,
    pub db_path: Option<String>,
    pub hot_db_path: Option<String>,
    pub log_level: String,
    pub enable_metrics: bool,
    /// Node ID (optional, will be generated if not provided)
    pub node_id: Option<NodeId>,
    /// Tor configuration
    pub tor: TorConfig,
    /// v8.9.x: AIOC service auth shared secret (HMAC-SHA256)
    /// Set via Q_AIOC_SERVICE_SECRET env var. When set, allows AIOC on localhost
    /// to call authenticated endpoints on behalf of logged-in wallet users.
    pub aioc_service_secret: Option<String>,

    // v0.0.22-beta: Quick Wins #1, #2, #4
    /// Allow manual block trigger endpoint (default: false for security)
    #[serde(default)]
    pub allow_manual_trigger: bool,

    /// Block production interval in seconds (5-300 range enforced)
    #[serde(default = "default_block_interval")]
    pub block_interval_secs: u64,

    /// Minimum solutions per block before production
    #[serde(default = "default_min_solutions")]
    pub min_solutions_per_block: usize,

    /// Maximum solutions per block (hard limit)
    #[serde(default = "default_max_solutions")]
    pub max_solutions_per_block: usize,

    /// Validator index (0-based, must be unique per validator)
    #[serde(default)]
    pub validator_index: u64,

    /// Total number of validators in network
    #[serde(default = "default_total_validators")]
    pub total_validators: u64,
}

fn default_block_interval() -> u64 {
    0
} // v8.0.9: 0 = produce as fast as possible (target 10+ bps with multiple producers)
fn default_min_solutions() -> usize {
    1
}
fn default_max_solutions() -> usize {
    10_000 // v8.6.0: Harmonized with Default impl (was 100, but Default::default() used 10_000)
}
fn default_total_validators() -> u64 {
    1
}

/// Tor-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorConfig {
    /// Enable Tor networking
    pub enabled: bool,
    /// Number of dedicated circuits
    pub circuit_count: usize,
    /// Tor onion service port
    pub onion_port: u16,
    /// Tor data directory
    pub data_dir: Option<PathBuf>,
    /// Tor-only mode (no fallback to direct connections)
    pub tor_only: bool,
    /// Enable Dandelion++ for traffic analysis resistance
    pub enable_dandelion: bool,
    /// Latency target in milliseconds
    pub latency_target_ms: u16,
    /// Bootstrap onion addresses
    pub bootstrap_onions: Vec<String>,
    /// SOCKS5 proxy address
    pub socks5_addr: Option<String>,
}

impl Default for TorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            circuit_count: 4,
            onion_port: 4001,
            data_dir: Some(PathBuf::from("/var/lib/qnk/tor")),
            tor_only: false,
            enable_dandelion: true,
            latency_target_ms: 300,
            bootstrap_onions: vec!["bootstrap.qnk.onion:4001".to_string()],
            socks5_addr: Some("127.0.0.1:9050".to_string()),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "0.0.0.0".to_string(),
            is_validator: true, // ✅ Default TRUE so localhost mining works out-of-the-box
            p2p_port: 8081,
            bootstrap_peers: vec![],
            database_url: None,
            db_path: None,
            hot_db_path: None,
            log_level: "info".to_string(),
            enable_metrics: true,
            node_id: None,
            tor: TorConfig::default(),
            aioc_service_secret: None,
            allow_manual_trigger: false,  // v0.0.22-beta: default secure
            block_interval_secs: 0, // v8.0.10: 0 = produce as fast as possible (10+ BPS target)
            min_solutions_per_block: 1, // v0.0.22-beta: default 1
            max_solutions_per_block: 10_000, // v7.1.4: Increased from 100 - queue backlog fix
            validator_index: 0,     // v0.0.22-beta: default primary
            total_validators: 1,    // v0.0.22-beta: default single validator
        }
    }
}

impl Config {
    /// Validate configuration on startup (v0.0.22-beta Quick Win #2)
    /// Returns error if configuration is invalid or unsafe
    pub fn validate(&self) -> anyhow::Result<()> {
        use tracing::{info, warn};

        // v8.0.10: Allow interval=0 for maximum block throughput (10+ BPS target)
        if self.block_interval_secs == 0 {
            info!("⚡ Maximum block production enabled: interval=0 (produce as fast as possible)");
        } else if self.block_interval_secs < 5 {
            warn!("⚡ Fast block production enabled: {} second intervals",
                  self.block_interval_secs);
        }

        if self.block_interval_secs > 300 {
            anyhow::bail!(
                "INVALID CONFIG: block_interval_secs ({}) must be <= 300 seconds (ensure liveness)",
                self.block_interval_secs
            );
        }

        // Validate solution limits
        if self.min_solutions_per_block > self.max_solutions_per_block {
            anyhow::bail!(
                "INVALID CONFIG: min_solutions_per_block ({}) must be <= max_solutions_per_block ({})",
                self.min_solutions_per_block,
                self.max_solutions_per_block
            );
        }

        if self.max_solutions_per_block > 50_000 {
            anyhow::bail!(
                "INVALID CONFIG: max_solutions_per_block ({}) must be <= 50000 (prevent DoS)",
                self.max_solutions_per_block
            );
        }

        // Validate port
        if self.port == 0 {
            anyhow::bail!("INVALID CONFIG: port must be specified");
        }

        // Validate validator configuration
        if self.is_validator {
            if self.validator_index >= self.total_validators {
                anyhow::bail!(
                    "INVALID CONFIG: validator_index ({}) must be < total_validators ({})",
                    self.validator_index,
                    self.total_validators
                );
            }

            if self.total_validators == 0 {
                anyhow::bail!("INVALID CONFIG: total_validators must be > 0");
            }

            // Warn if multi-validator without coordination
            if self.total_validators > 1 {
                warn!(
                    "⚠️  Multi-validator mode enabled ({} validators)",
                    self.total_validators
                );
                warn!("⚠️  Ensure all validators have identical total_validators setting");
                warn!(
                    "⚠️  Ensure each validator has unique validator_index (0 to {})",
                    self.total_validators - 1
                );
                warn!("⚠️  Simple coordination: Only validator 0 produces empty blocks");
            } else {
                info!("✅ Single validator mode (default)");
            }
        }

        // Warn about manual trigger
        if self.allow_manual_trigger {
            warn!("⚠️  Manual block trigger is ENABLED");
            warn!("⚠️  This should only be used for testing/development");
            warn!("⚠️  Ensure API authentication is configured!");
        }

        // Success
        info!("✅ Configuration validated successfully");
        Ok(())
    }

    pub fn from_env() -> anyhow::Result<Self> {
        let mut config = Self::default();

        if let Ok(port) = env::var("Q_API_PORT") {
            config.port = port.parse()?;
        }

        if let Ok(host) = env::var("Q_API_HOST") {
            config.host = host;
        }

        if let Ok(is_validator) = env::var("Q_IS_VALIDATOR") {
            config.is_validator = is_validator.parse().unwrap_or(false);
        }

        if let Ok(p2p_port) = env::var("Q_P2P_PORT") {
            config.p2p_port = p2p_port.parse()?;
        }

        // Bootstrap peer discovery - automatic and manual
        if let Ok(bootstrap_peers) = env::var("Q_BOOTSTRAP_PEERS") {
            config.bootstrap_peers = bootstrap_peers
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        // Automatic bootstrap discovery via HTTP API
        if config.bootstrap_peers.is_empty() {
            use tracing::{info, warn};

            // Default bootstrap URL - Q-NarwhalKnight Testnet Masternode
            let bootstrap_url = env::var("Q_BOOTSTRAP_URL")
                .unwrap_or_else(|_| "https://quillon.xyz".to_string());

            info!(
                "🔍 Attempting automatic bootstrap discovery from {}",
                bootstrap_url
            );

            match Self::fetch_bootstrap_peers(&bootstrap_url) {
                Ok(peers) => {
                    if !peers.is_empty() {
                        info!(
                            "✅ Discovered {} bootstrap peer(s) automatically",
                            peers.len()
                        );
                        for peer in &peers {
                            info!("   📡 {}", peer);
                        }
                        config.bootstrap_peers = peers;
                    } else {
                        warn!("⚠️  No bootstrap peers discovered from {}", bootstrap_url);
                        warn!("⚠️  Using hardcoded fallback bootstrap peers");
                        config.bootstrap_peers = Self::hardcoded_bootstrap_peers();
                    }
                }
                Err(e) => {
                    warn!(
                        "⚠️  Failed to fetch bootstrap peers from {}: {}",
                        bootstrap_url, e
                    );
                    warn!("⚠️  Using hardcoded fallback bootstrap peers");
                    config.bootstrap_peers = Self::hardcoded_bootstrap_peers();
                }
            }
        }

        if let Ok(database_url) = env::var("DATABASE_URL") {
            config.database_url = Some(database_url);
        }

        if let Ok(db_path) = env::var("Q_DB_PATH") {
            config.db_path = Some(db_path);
        }

        if let Ok(hot_db_path) = env::var("Q_HOT_DB_PATH") {
            config.hot_db_path = Some(hot_db_path);
        }

        if let Ok(log_level) = env::var("Q_LOG_LEVEL") {
            config.log_level = log_level;
        }

        if let Ok(enable_metrics) = env::var("Q_ENABLE_METRICS") {
            config.enable_metrics = enable_metrics.parse().unwrap_or(true);
        }

        // Tor configuration
        if let Ok(tor_enabled) = env::var("Q_TOR_ENABLED") {
            config.tor.enabled = tor_enabled.parse().unwrap_or(false);
        }

        if let Ok(tor_circuit_count) = env::var("Q_TOR_CIRCUIT_COUNT") {
            config.tor.circuit_count = tor_circuit_count.parse().unwrap_or(4);
        }

        if let Ok(tor_onion_port) = env::var("Q_TOR_ONION_PORT") {
            config.tor.onion_port = tor_onion_port.parse().unwrap_or(4001);
        }

        if let Ok(tor_data_dir) = env::var("Q_TOR_DATA_DIR") {
            config.tor.data_dir = Some(PathBuf::from(tor_data_dir));
        }

        if let Ok(tor_only) = env::var("Q_TOR_ONLY") {
            config.tor.tor_only = tor_only.parse().unwrap_or(false);
        }

        if let Ok(tor_dandelion) = env::var("Q_TOR_DANDELION") {
            config.tor.enable_dandelion = tor_dandelion.parse().unwrap_or(true);
        }

        if let Ok(tor_latency) = env::var("Q_TOR_LATENCY_TARGET_MS") {
            config.tor.latency_target_ms = tor_latency.parse().unwrap_or(300);
        }

        if let Ok(tor_bootstrap) = env::var("Q_TOR_BOOTSTRAP_ONIONS") {
            config.tor.bootstrap_onions = tor_bootstrap
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        if let Ok(tor_socks5) = env::var("Q_TOR_SOCKS5_ADDR") {
            config.tor.socks5_addr = Some(tor_socks5);
        }

        // v0.0.22-beta: Block production configuration (Quick Wins)
        if let Ok(allow_manual_trigger) = env::var("Q_ALLOW_MANUAL_TRIGGER") {
            config.allow_manual_trigger = allow_manual_trigger.parse().unwrap_or(false);
        }

        if let Ok(block_interval) = env::var("Q_BLOCK_INTERVAL_SECS") {
            config.block_interval_secs = block_interval.parse().unwrap_or(15);
        }

        if let Ok(min_solutions) = env::var("Q_MIN_SOLUTIONS_PER_BLOCK") {
            config.min_solutions_per_block = min_solutions.parse().unwrap_or(1);
        }

        if let Ok(max_solutions) = env::var("Q_MAX_SOLUTIONS_PER_BLOCK") {
            config.max_solutions_per_block = max_solutions.parse().unwrap_or(10_000);
        }

        if let Ok(validator_index) = env::var("Q_VALIDATOR_INDEX") {
            config.validator_index = validator_index.parse().unwrap_or(0);
        }

        if let Ok(total_validators) = env::var("Q_TOTAL_VALIDATORS") {
            config.total_validators = total_validators.parse().unwrap_or(1);
        }

        // v8.9.x: AIOC service auth
        if let Ok(aioc_secret) = env::var("Q_AIOC_SERVICE_SECRET") {
            if !aioc_secret.is_empty() {
                config.aioc_service_secret = Some(aioc_secret);
            }
        }

        Ok(config)
    }

    /// v10.0.8: Hardcoded fallback bootstrap peers for when HTTP discovery fails.
    /// This is critical for Windows where:
    /// 1. HTTPS may fail due to TLS backend issues (native-tls/schannel)
    /// 2. mDNS is disabled on Windows
    /// Without these fallbacks, Windows nodes get zero peers and never sync.
    fn hardcoded_bootstrap_peers() -> Vec<String> {
        use tracing::info;
        let peers = vec![
            // Epsilon (10Gbit SUPERNODE — primary sync target)
            "/ip4/89.149.241.126/tcp/9001/p2p/12D3KooWFpbXxxZJQ4FX9FGXrE5vaeNTCnZmLn6bqToRCMuiMpxM".to_string(),
            // DNS fallback — bypasses per-IP HTTP verification (no timeout on firewalled port 8080)
            "/dns4/quillon.xyz/tcp/9001/p2p/12D3KooWFpbXxxZJQ4FX9FGXrE5vaeNTCnZmLn6bqToRCMuiMpxM".to_string(),
            // Beta (production bootstrap) — updated peer ID
            "/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWKyjQUYXJQ8y8WdHbtMVxsNt4a412Ccqdr1oKjSY8fy93".to_string(),
            // Gamma (backup bootstrap)
            "/ip4/109.205.176.60/tcp/9001/p2p/12D3KooWFfZKfKbBnB5SehTRBacHndyhJ6aQWxTAQrrwXA7761cH".to_string(),
        ];
        info!("📡 Loaded {} hardcoded fallback bootstrap peers", peers.len());
        for peer in &peers {
            info!("   📡 {}", peer);
        }
        peers
    }

    /// Fetch bootstrap peers from a remote API endpoint
    ///
    /// This enables automatic peer discovery by querying the masternode's
    /// /api/v1/status endpoint and extracting libp2p listen addresses.
    ///
    /// # Arguments
    /// * `url` - Base URL of the bootstrap node (e.g., "http://185.182.185.227:8080")
    ///
    /// # Returns
    /// * `Ok(Vec<String>)` - List of bootstrap peer multiaddrs
    /// * `Err(anyhow::Error)` - If the request fails or response is invalid
    fn fetch_bootstrap_peers(url: &str) -> anyhow::Result<Vec<String>> {
        use std::time::Duration;

        // Build the status endpoint URL
        let status_url = if url.ends_with('/') {
            format!("{}api/v1/status", url)
        } else {
            format!("{}/api/v1/status", url)
        };

        // Create a blocking HTTP client with timeout
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;

        // Fetch the status endpoint
        let response = client.get(&status_url).send()?;

        if !response.status().is_success() {
            anyhow::bail!("HTTP error: {}", response.status());
        }

        // Parse the JSON response
        let json: serde_json::Value = response.json()?;

        // Extract multiaddrs from the response (v0.9.5-beta fix)
        // Server returns: { "data": { "multiaddrs": [...] } }
        let addrs = json
            .get("data")
            .and_then(|data| data.get("multiaddrs"))
            .and_then(|addrs| addrs.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .filter(|s| !s.contains("127.0.0.1") && !s.contains("::1")) // Skip localhost
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_else(Vec::new);

        Ok(addrs)
    }
}
