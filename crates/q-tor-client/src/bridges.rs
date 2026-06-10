/// Bridge Support for Q-NarwhalKnight
///
/// Bridges are Tor relays that are not listed in the main Tor directory.
/// They help users in censored regions connect to the Tor network by
/// providing entry points that are harder for censors to discover and block.
///
/// # Pluggable Transports
/// Pluggable transports disguise Tor traffic to look like other protocols:
/// - obfs4: Obfuscation protocol (recommended)
/// - meek: Domain fronting through CDNs
/// - snowflake: WebRTC-based transport through browser proxies
/// - webtunnel: HTTPS-based transport
///
/// # Threat Model
/// In heavily censored regions (China, Iran, Russia), censors can:
/// - Block known Tor relay IPs
/// - Detect Tor protocol signatures (DPI)
/// - Block based on traffic patterns
///
/// Bridges + pluggable transports counter all three.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    net::{IpAddr, SocketAddr},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Pluggable transport types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransportType {
    /// obfs4 - The recommended pluggable transport
    /// Provides encryption and obfuscation
    Obfs4,
    /// meek - Domain fronting through CDNs (Azure, Amazon, Google)
    /// Very hard to block but slow
    Meek,
    /// Snowflake - WebRTC-based transport through volunteer browser proxies
    /// Decentralized and hard to block
    Snowflake,
    /// webtunnel - Disguises Tor as HTTPS traffic
    WebTunnel,
    /// scramblesuit - Older obfuscation protocol
    ScrambleSuit,
    /// Direct connection (no pluggable transport)
    Direct,
}

impl TransportType {
    pub fn name(&self) -> &'static str {
        match self {
            TransportType::Obfs4 => "obfs4",
            TransportType::Meek => "meek",
            TransportType::Snowflake => "snowflake",
            TransportType::WebTunnel => "webtunnel",
            TransportType::ScrambleSuit => "scramblesuit",
            TransportType::Direct => "direct",
        }
    }

    /// Get the default port for this transport
    pub fn default_port(&self) -> u16 {
        match self {
            TransportType::Obfs4 => 443,      // HTTPS port for stealth
            TransportType::Meek => 443,       // HTTPS
            TransportType::Snowflake => 443,  // WebRTC
            TransportType::WebTunnel => 443,  // HTTPS
            TransportType::ScrambleSuit => 443,
            TransportType::Direct => 9001,    // Standard Tor OR port
        }
    }

    /// Whether this transport requires additional parameters
    pub fn requires_params(&self) -> bool {
        matches!(self, TransportType::Obfs4 | TransportType::Meek | TransportType::WebTunnel)
    }

    /// Resistance level against DPI (Deep Packet Inspection)
    pub fn dpi_resistance(&self) -> DpiResistance {
        match self {
            TransportType::Obfs4 => DpiResistance::High,
            TransportType::Meek => DpiResistance::VeryHigh,
            TransportType::Snowflake => DpiResistance::High,
            TransportType::WebTunnel => DpiResistance::VeryHigh,
            TransportType::ScrambleSuit => DpiResistance::Medium,
            TransportType::Direct => DpiResistance::None,
        }
    }

    /// Typical latency overhead
    pub fn latency_overhead(&self) -> Duration {
        match self {
            TransportType::Obfs4 => Duration::from_millis(50),
            TransportType::Meek => Duration::from_millis(500),      // Domain fronting is slow
            TransportType::Snowflake => Duration::from_millis(200), // WebRTC overhead
            TransportType::WebTunnel => Duration::from_millis(100),
            TransportType::ScrambleSuit => Duration::from_millis(50),
            TransportType::Direct => Duration::ZERO,
        }
    }
}

/// DPI (Deep Packet Inspection) resistance level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DpiResistance {
    None,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// A bridge relay configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Transport type
    pub transport: TransportType,
    /// Bridge address (IP:port)
    pub address: SocketAddr,
    /// Bridge fingerprint (40 hex chars)
    pub fingerprint: String,
    /// Transport-specific parameters
    pub params: HashMap<String, String>,
    /// Human-readable nickname (optional)
    pub nickname: Option<String>,
}

impl BridgeConfig {
    /// Create a new obfs4 bridge
    pub fn obfs4(
        address: SocketAddr,
        fingerprint: String,
        cert: String,
        iat_mode: u8,
    ) -> Self {
        let mut params = HashMap::new();
        params.insert("cert".to_string(), cert);
        params.insert("iat-mode".to_string(), iat_mode.to_string());

        Self {
            transport: TransportType::Obfs4,
            address,
            fingerprint,
            params,
            nickname: None,
        }
    }

    /// Create a meek bridge
    pub fn meek(address: SocketAddr, fingerprint: String, url: String, front: String) -> Self {
        let mut params = HashMap::new();
        params.insert("url".to_string(), url);
        params.insert("front".to_string(), front);

        Self {
            transport: TransportType::Meek,
            address,
            fingerprint,
            params,
            nickname: None,
        }
    }

    /// Create a snowflake bridge
    pub fn snowflake(fingerprint: String, broker_url: Option<String>) -> Self {
        let mut params = HashMap::new();
        if let Some(url) = broker_url {
            params.insert("url".to_string(), url);
        }
        // Use default snowflake broker
        params.insert("fronts".to_string(), "www.google.com,ajax.aspnetcdn.com".to_string());
        params.insert("ice".to_string(), "stun:stun.l.google.com:19302".to_string());

        Self {
            transport: TransportType::Snowflake,
            address: "192.0.2.1:1".parse().unwrap(), // Placeholder, snowflake doesn't use direct IP
            fingerprint,
            params,
            nickname: Some("snowflake".to_string()),
        }
    }

    /// Create a webtunnel bridge
    pub fn webtunnel(address: SocketAddr, fingerprint: String, url: String) -> Self {
        let mut params = HashMap::new();
        params.insert("url".to_string(), url);

        Self {
            transport: TransportType::WebTunnel,
            address,
            fingerprint,
            params,
            nickname: None,
        }
    }

    /// Create a direct (non-PT) bridge
    pub fn direct(address: SocketAddr, fingerprint: String) -> Self {
        Self {
            transport: TransportType::Direct,
            address,
            fingerprint,
            params: HashMap::new(),
            nickname: None,
        }
    }

    /// Parse a bridge line (format used by BridgeDB)
    pub fn from_bridge_line(line: &str) -> Result<Self> {
        // Format: [transport] IP:PORT [fingerprint] [params...]
        // Example: obfs4 192.0.2.1:443 FINGERPRINT cert=... iat-mode=0

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Err(anyhow!("Empty bridge line"));
        }

        let (transport, rest) = if parts[0].contains(':') {
            // No transport specified, direct connection
            (TransportType::Direct, &parts[..])
        } else {
            let transport = match parts[0].to_lowercase().as_str() {
                "obfs4" => TransportType::Obfs4,
                "meek" | "meek_lite" => TransportType::Meek,
                "snowflake" => TransportType::Snowflake,
                "webtunnel" => TransportType::WebTunnel,
                "scramblesuit" => TransportType::ScrambleSuit,
                _ => return Err(anyhow!("Unknown transport: {}", parts[0])),
            };
            (transport, &parts[1..])
        };

        if rest.is_empty() {
            return Err(anyhow!("Missing address in bridge line"));
        }

        let address: SocketAddr = rest[0]
            .parse()
            .map_err(|e| anyhow!("Invalid address: {}", e))?;

        let fingerprint = if rest.len() > 1 {
            rest[1].to_string()
        } else {
            String::new()
        };

        let mut params = HashMap::new();
        for part in rest.iter().skip(2) {
            if let Some((key, value)) = part.split_once('=') {
                params.insert(key.to_string(), value.to_string());
            }
        }

        Ok(Self {
            transport,
            address,
            fingerprint,
            params,
            nickname: None,
        })
    }

    /// Convert to bridge line format
    pub fn to_bridge_line(&self) -> String {
        let mut parts = vec![];

        if self.transport != TransportType::Direct {
            parts.push(self.transport.name().to_string());
        }

        parts.push(self.address.to_string());

        if !self.fingerprint.is_empty() {
            parts.push(self.fingerprint.clone());
        }

        for (key, value) in &self.params {
            parts.push(format!("{}={}", key, value));
        }

        parts.join(" ")
    }
}

/// Bridge health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeHealth {
    /// Bridge fingerprint
    pub fingerprint: String,
    /// Last successful connection (not serialized - Instant doesn't impl serde)
    #[serde(skip)]
    pub last_success: Option<Instant>,
    /// Last failure (not serialized - Instant doesn't impl serde)
    #[serde(skip)]
    pub last_failure: Option<Instant>,
    /// Success count
    pub success_count: u64,
    /// Failure count
    pub failure_count: u64,
    /// Average latency
    pub avg_latency_ms: f64,
    /// Currently reachable
    pub is_reachable: bool,
}

impl BridgeHealth {
    pub fn new(fingerprint: String) -> Self {
        Self {
            fingerprint,
            last_success: None,
            last_failure: None,
            success_count: 0,
            failure_count: 0,
            avg_latency_ms: 0.0,
            is_reachable: false,
        }
    }

    pub fn record_success(&mut self, latency_ms: f64) {
        self.success_count += 1;
        self.last_success = Some(Instant::now());
        self.is_reachable = true;

        // Update average latency
        let total_ms = self.avg_latency_ms * (self.success_count - 1) as f64 + latency_ms;
        self.avg_latency_ms = total_ms / self.success_count as f64;
    }

    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure = Some(Instant::now());
        self.is_reachable = false;
    }

    /// Calculate reliability score (0.0 - 1.0)
    pub fn reliability(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            return 0.0;
        }
        self.success_count as f64 / total as f64
    }
}

/// Configuration for bridge management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgesConfig {
    /// Enable bridge usage
    pub enabled: bool,
    /// Preferred transport type
    pub preferred_transport: TransportType,
    /// List of configured bridges
    pub bridges: Vec<BridgeConfig>,
    /// Auto-fetch bridges from BridgeDB
    pub auto_fetch: bool,
    /// BridgeDB URL
    pub bridgedb_url: Option<String>,
    /// Fallback to direct connection if all bridges fail
    pub fallback_to_direct: bool,
    /// Maximum bridges to try before giving up
    pub max_attempts: usize,
    /// Bridge test interval
    pub test_interval: Duration,
}

impl Default for BridgesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            preferred_transport: TransportType::Obfs4,
            bridges: Vec::new(),
            auto_fetch: true,
            bridgedb_url: Some("https://bridges.torproject.org/bridges".to_string()),
            fallback_to_direct: true,
            max_attempts: 5,
            test_interval: Duration::from_secs(3600), // Test bridges every hour
        }
    }
}

impl BridgesConfig {
    /// Create configuration for heavily censored regions (China, Iran)
    pub fn high_censorship() -> Self {
        Self {
            enabled: true,
            preferred_transport: TransportType::Obfs4,
            bridges: Vec::new(),
            auto_fetch: true,
            bridgedb_url: Some("https://bridges.torproject.org/bridges".to_string()),
            fallback_to_direct: false, // Never try direct in heavy censorship
            max_attempts: 10,
            test_interval: Duration::from_secs(1800), // Test more frequently
        }
    }

    /// Create configuration with snowflake (for extreme censorship)
    pub fn snowflake_mode() -> Self {
        // Default Snowflake bridge config
        let snowflake = BridgeConfig::snowflake(
            "2B280B23E1107BB62ABFC40DDCC8824814F80A72".to_string(),
            Some("https://snowflake-broker.torproject.net/".to_string()),
        );

        Self {
            enabled: true,
            preferred_transport: TransportType::Snowflake,
            bridges: vec![snowflake],
            auto_fetch: false, // Snowflake is self-contained
            bridgedb_url: None,
            fallback_to_direct: false,
            max_attempts: 3,
            test_interval: Duration::from_secs(600),
        }
    }

    /// Create configuration with meek (domain fronting)
    pub fn meek_mode() -> Self {
        // Default meek-azure bridge
        let meek = BridgeConfig::meek(
            "0.0.0.1:1".parse().unwrap(), // Meek doesn't use direct IP
            "97700DFE9F483596DDA6264C4D7DF7641E1E39CE".to_string(),
            "https://meek.azureedge.net/".to_string(),
            "ajax.aspnetcdn.com".to_string(),
        );

        Self {
            enabled: true,
            preferred_transport: TransportType::Meek,
            bridges: vec![meek],
            auto_fetch: false,
            bridgedb_url: None,
            fallback_to_direct: false,
            max_attempts: 3,
            test_interval: Duration::from_secs(600),
        }
    }
}

/// Bridge manager
pub struct BridgeManager {
    /// Configuration
    config: BridgesConfig,
    /// Bridge health tracking
    health: RwLock<HashMap<String, BridgeHealth>>,
    /// Currently active bridge
    active_bridge: RwLock<Option<BridgeConfig>>,
    /// Failed bridge fingerprints (temporary blacklist)
    failed_bridges: RwLock<HashSet<String>>,
    /// Last bridge test time
    last_test: RwLock<Option<Instant>>,
}

impl BridgeManager {
    /// Create a new bridge manager
    pub fn new(config: BridgesConfig) -> Self {
        info!(
            "🌉 Initializing bridge manager: transport={}, bridges={}",
            config.preferred_transport.name(),
            config.bridges.len()
        );

        Self {
            config,
            health: RwLock::new(HashMap::new()),
            active_bridge: RwLock::new(None),
            failed_bridges: RwLock::new(HashSet::new()),
            last_test: RwLock::new(None),
        }
    }

    /// Add a bridge
    pub async fn add_bridge(&self, bridge: BridgeConfig) {
        info!(
            "🌉 Adding {} bridge: {}",
            bridge.transport.name(),
            bridge.address
        );

        // Initialize health tracking
        let mut health = self.health.write().await;
        health.insert(
            bridge.fingerprint.clone(),
            BridgeHealth::new(bridge.fingerprint.clone()),
        );
    }

    /// Get the best available bridge
    pub async fn get_best_bridge(&self) -> Option<BridgeConfig> {
        // First, check if we have an active healthy bridge
        {
            let active = self.active_bridge.read().await;
            if let Some(bridge) = active.as_ref() {
                let health = self.health.read().await;
                if let Some(h) = health.get(&bridge.fingerprint) {
                    if h.is_reachable {
                        return Some(bridge.clone());
                    }
                }
            }
        }

        // Find the best bridge by reliability and preferred transport
        let failed = self.failed_bridges.read().await;
        let health = self.health.read().await;

        let mut best: Option<(BridgeConfig, f64)> = None;

        for bridge in &self.config.bridges {
            // Skip failed bridges
            if failed.contains(&bridge.fingerprint) {
                continue;
            }

            // Calculate score
            let mut score = 0.0;

            // Prefer the configured transport type
            if bridge.transport == self.config.preferred_transport {
                score += 1.0;
            }

            // Add reliability score
            if let Some(h) = health.get(&bridge.fingerprint) {
                score += h.reliability();
            }

            // Add DPI resistance score
            score += match bridge.transport.dpi_resistance() {
                DpiResistance::VeryHigh => 0.5,
                DpiResistance::High => 0.3,
                DpiResistance::Medium => 0.2,
                DpiResistance::Low => 0.1,
                DpiResistance::None => 0.0,
            };

            if best.is_none() || score > best.as_ref().unwrap().1 {
                best = Some((bridge.clone(), score));
            }
        }

        if let Some((bridge, _)) = best {
            // Set as active bridge
            let mut active = self.active_bridge.write().await;
            *active = Some(bridge.clone());
            return Some(bridge);
        }

        None
    }

    /// Record bridge connection success
    pub async fn record_success(&self, fingerprint: &str, latency_ms: f64) {
        let mut health = self.health.write().await;
        if let Some(h) = health.get_mut(fingerprint) {
            h.record_success(latency_ms);
        }

        // Remove from failed list if present
        let mut failed = self.failed_bridges.write().await;
        failed.remove(fingerprint);

        debug!(
            "🌉 Bridge {} succeeded (latency: {:.0}ms)",
            &fingerprint[..8],
            latency_ms
        );
    }

    /// Record bridge connection failure
    pub async fn record_failure(&self, fingerprint: &str) {
        let mut health = self.health.write().await;
        if let Some(h) = health.get_mut(fingerprint) {
            h.record_failure();
        }

        warn!("🌉 Bridge {} failed", &fingerprint[..8]);

        // Check if we should blacklist
        if let Some(h) = health.get(fingerprint) {
            if h.failure_count >= 3 && h.reliability() < 0.5 {
                let mut failed = self.failed_bridges.write().await;
                failed.insert(fingerprint.to_string());
                warn!("🌉 Bridge {} blacklisted due to repeated failures", &fingerprint[..8]);

                // Clear active bridge if it was blacklisted
                let mut active = self.active_bridge.write().await;
                if active.as_ref().map(|b| b.fingerprint == fingerprint).unwrap_or(false) {
                    *active = None;
                }
            }
        }
    }

    /// Test all bridges
    pub async fn test_bridges(&self) -> BridgeTestResults {
        info!("🌉 Testing {} bridges", self.config.bridges.len());

        let mut results = BridgeTestResults {
            tested: 0,
            reachable: 0,
            unreachable: 0,
            best_latency_ms: f64::MAX,
            tested_at: Instant::now(),
        };

        for bridge in &self.config.bridges {
            results.tested += 1;

            // Simulate bridge test (in production, actually connect)
            let success = self.test_single_bridge(&bridge).await;

            if success {
                results.reachable += 1;
            } else {
                results.unreachable += 1;
            }
        }

        // Update last test time
        {
            let mut last_test = self.last_test.write().await;
            *last_test = Some(Instant::now());
        }

        info!(
            "🌉 Bridge test complete: {}/{} reachable",
            results.reachable, results.tested
        );

        results
    }

    /// Test a single bridge
    async fn test_single_bridge(&self, bridge: &BridgeConfig) -> bool {
        debug!(
            "🌉 Testing {} bridge at {}",
            bridge.transport.name(),
            bridge.address
        );

        // In production, this would actually attempt a connection
        // For now, simulate based on transport type
        let success = match bridge.transport {
            TransportType::Snowflake => true, // Snowflake usually works
            TransportType::Meek => true,      // Meek usually works (domain fronting)
            TransportType::Obfs4 => true,     // Depends on specific bridge
            _ => true,
        };

        if success {
            // Simulate latency
            let latency = bridge.transport.latency_overhead().as_millis() as f64
                + rand::random::<f64>() * 100.0;
            self.record_success(&bridge.fingerprint, latency).await;
        } else {
            self.record_failure(&bridge.fingerprint).await;
        }

        success
    }

    /// Clear the failed bridges blacklist
    pub async fn clear_blacklist(&self) {
        let mut failed = self.failed_bridges.write().await;
        failed.clear();
        info!("🌉 Cleared bridge blacklist");
    }

    /// Get bridge status summary
    pub async fn get_status(&self) -> BridgeStatus {
        let health = self.health.read().await;
        let failed = self.failed_bridges.read().await;
        let active = self.active_bridge.read().await;

        let reachable_count = health.values().filter(|h| h.is_reachable).count();

        BridgeStatus {
            enabled: self.config.enabled,
            total_bridges: self.config.bridges.len(),
            reachable_bridges: reachable_count,
            blacklisted_bridges: failed.len(),
            active_bridge: active.as_ref().map(|b| b.fingerprint.clone()),
            active_transport: active.as_ref().map(|b| b.transport),
            preferred_transport: self.config.preferred_transport,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &BridgesConfig {
        &self.config
    }
}

/// Bridge test results
#[derive(Debug, Clone)]
pub struct BridgeTestResults {
    pub tested: usize,
    pub reachable: usize,
    pub unreachable: usize,
    pub best_latency_ms: f64,
    pub tested_at: Instant,
}

/// Bridge status summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeStatus {
    pub enabled: bool,
    pub total_bridges: usize,
    pub reachable_bridges: usize,
    pub blacklisted_bridges: usize,
    pub active_bridge: Option<String>,
    pub active_transport: Option<TransportType>,
    pub preferred_transport: TransportType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bridge_line() {
        let line = "obfs4 192.0.2.1:443 FINGERPRINT cert=XXXX iat-mode=0";
        let bridge = BridgeConfig::from_bridge_line(line).unwrap();

        assert_eq!(bridge.transport, TransportType::Obfs4);
        assert_eq!(bridge.address.port(), 443);
        assert_eq!(bridge.fingerprint, "FINGERPRINT");
        assert_eq!(bridge.params.get("cert"), Some(&"XXXX".to_string()));
    }

    #[test]
    fn test_bridge_line_roundtrip() {
        let bridge = BridgeConfig::obfs4(
            "192.0.2.1:443".parse().unwrap(),
            "FINGERPRINT".to_string(),
            "CERTDATA".to_string(),
            0,
        );

        let line = bridge.to_bridge_line();
        assert!(line.contains("obfs4"));
        assert!(line.contains("192.0.2.1:443"));
        assert!(line.contains("FINGERPRINT"));
    }

    #[test]
    fn test_transport_properties() {
        assert_eq!(TransportType::Obfs4.dpi_resistance(), DpiResistance::High);
        assert_eq!(TransportType::Meek.dpi_resistance(), DpiResistance::VeryHigh);
        assert_eq!(TransportType::Direct.dpi_resistance(), DpiResistance::None);
    }

    #[test]
    fn test_bridge_health() {
        let mut health = BridgeHealth::new("test".to_string());

        health.record_success(100.0);
        health.record_success(200.0);
        health.record_failure();

        assert_eq!(health.success_count, 2);
        assert_eq!(health.failure_count, 1);
        assert!((health.reliability() - 0.666).abs() < 0.01);
        assert!((health.avg_latency_ms - 150.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_bridge_manager() {
        let config = BridgesConfig::snowflake_mode();
        let manager = BridgeManager::new(config);

        let status = manager.get_status().await;
        assert!(status.enabled);
        assert_eq!(status.preferred_transport, TransportType::Snowflake);
    }
}
