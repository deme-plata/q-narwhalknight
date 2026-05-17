use tracing::{debug, info, warn, error};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use tokio::time::interval;
use std::sync::{Arc, Mutex};

/// Comprehensive debugging system for peer discovery and connection issues
#[derive(Debug, Clone)]
pub struct DiscoveryDebugger {
    pub node_id: String,
    pub discovery_attempts: Arc<Mutex<HashMap<String, DiscoveryAttempt>>>,
    pub connection_attempts: Arc<Mutex<HashMap<String, ConnectionAttempt>>>,
    pub network_stats: Arc<Mutex<NetworkDebugStats>>,
}

#[derive(Debug, Clone)]
pub struct DiscoveryAttempt {
    pub method: String, // "BEP-44", "DNS-Phantom", "Bitcoin-Bridge", "Production-Discovery"
    pub timestamp: u64,
    pub peer_id: Option<String>,
    pub onion_address: Option<String>,
    pub status: DiscoveryStatus,
    pub details: String,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ConnectionAttempt {
    pub target_peer: String,
    pub target_address: String,
    pub method: String, // "Tor", "Direct-TCP", "libp2p"
    pub timestamp: u64,
    pub status: ConnectionStatus,
    pub latency_ms: Option<u64>,
    pub error_message: Option<String>,
    pub connection_details: String,
}

#[derive(Debug, Clone)]
pub enum DiscoveryStatus {
    Started,
    PeersFound(u32),
    NopeersFound,
    Failed,
    Timeout,
}

#[derive(Debug, Clone)]
pub enum ConnectionStatus {
    Attempting,
    Connected,
    Failed,
    Timeout,
    Rejected,
}

#[derive(Debug, Clone)]
pub struct NetworkDebugStats {
    pub total_discovery_attempts: u32,
    pub successful_discoveries: u32,
    pub total_connection_attempts: u32,
    pub successful_connections: u32,
    pub active_peers: u32,
    pub last_successful_discovery: Option<u64>,
    pub last_successful_connection: Option<u64>,
    pub discovery_methods: HashMap<String, u32>,
    pub connection_methods: HashMap<String, u32>,
    pub error_patterns: HashMap<String, u32>,
}

impl DiscoveryDebugger {
    pub fn new(node_id: String) -> Self {
        info!("🔧 DEBUG: Initializing comprehensive discovery debugger for node: {}", node_id);

        Self {
            node_id,
            discovery_attempts: Arc::new(Mutex::new(HashMap::new())),
            connection_attempts: Arc::new(Mutex::new(HashMap::new())),
            network_stats: Arc::new(Mutex::new(NetworkDebugStats {
                total_discovery_attempts: 0,
                successful_discoveries: 0,
                total_connection_attempts: 0,
                successful_connections: 0,
                active_peers: 0,
                last_successful_discovery: None,
                last_successful_connection: None,
                discovery_methods: HashMap::new(),
                connection_methods: HashMap::new(),
                error_patterns: HashMap::new(),
            })),
        }
    }

    /// Log a discovery attempt with detailed debugging
    pub fn log_discovery_attempt(
        &self,
        method: &str,
        peer_id: Option<String>,
        onion_address: Option<String>,
        status: DiscoveryStatus,
        details: &str,
        error: Option<String>,
    ) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let attempt = DiscoveryAttempt {
            method: method.to_string(),
            timestamp,
            peer_id: peer_id.clone(),
            onion_address: onion_address.clone(),
            status: status.clone(),
            details: details.to_string(),
            error_message: error.clone(),
        };

        // Store attempt
        let attempt_id = format!("{}-{}-{}", method, timestamp,
            peer_id.as_deref().unwrap_or("unknown"));
        {
            let mut attempts = self.discovery_attempts.lock().unwrap();
            attempts.insert(attempt_id.clone(), attempt);
        }

        // Update stats
        {
            let mut stats = self.network_stats.lock().unwrap();
            stats.total_discovery_attempts += 1;
            *stats.discovery_methods.entry(method.to_string()).or_insert(0) += 1;

            if let DiscoveryStatus::PeersFound(count) = &status {
                if *count > 0 {
                    stats.successful_discoveries += 1;
                    stats.last_successful_discovery = Some(timestamp);
                }
            }

            if let Some(err) = &error {
                *stats.error_patterns.entry(err.clone()).or_insert(0) += 1;
            }
        }

        // Detailed logging based on status
        match status {
            DiscoveryStatus::Started => {
                info!("🔍 DEBUG DISCOVERY [{} - {}]: STARTED - {}",
                    method, self.node_id, details);
            }
            DiscoveryStatus::PeersFound(count) => {
                if count > 0 {
                    info!("✅ DEBUG DISCOVERY [{} - {}]: FOUND {} PEERS - {}",
                        method, self.node_id, count, details);
                    if let Some(peer) = &peer_id {
                        info!("🎯 DEBUG PEER: {} discovered via {} at address: {}",
                            peer, method, onion_address.as_deref().unwrap_or("unknown"));
                    }
                } else {
                    warn!("🔍 DEBUG DISCOVERY [{} - {}]: NO PEERS FOUND - {}",
                        method, self.node_id, details);
                }
            }
            DiscoveryStatus::NopeersFound => {
                warn!("⚠️  DEBUG DISCOVERY [{} - {}]: ZERO PEERS - {} (This indicates discovery is working but network is empty)",
                    method, self.node_id, details);
            }
            DiscoveryStatus::Failed => {
                error!("❌ DEBUG DISCOVERY [{} - {}]: FAILED - {} | Error: {}",
                    method, self.node_id, details, error.as_deref().unwrap_or("Unknown"));
            }
            DiscoveryStatus::Timeout => {
                warn!("⏰ DEBUG DISCOVERY [{} - {}]: TIMEOUT - {}",
                    method, self.node_id, details);
            }
        }
    }

    /// Log a connection attempt with detailed debugging
    pub fn log_connection_attempt(
        &self,
        target_peer: &str,
        target_address: &str,
        method: &str,
        status: ConnectionStatus,
        latency_ms: Option<u64>,
        error: Option<String>,
        details: &str,
    ) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let attempt = ConnectionAttempt {
            target_peer: target_peer.to_string(),
            target_address: target_address.to_string(),
            method: method.to_string(),
            timestamp,
            status: status.clone(),
            latency_ms,
            error_message: error.clone(),
            connection_details: details.to_string(),
        };

        // Store attempt
        let attempt_id = format!("{}-{}-{}", target_peer, method, timestamp);
        {
            let mut attempts = self.connection_attempts.lock().unwrap();
            attempts.insert(attempt_id, attempt);
        }

        // Update stats
        {
            let mut stats = self.network_stats.lock().unwrap();
            stats.total_connection_attempts += 1;
            *stats.connection_methods.entry(method.to_string()).or_insert(0) += 1;

            if let ConnectionStatus::Connected = &status {
                stats.successful_connections += 1;
                stats.last_successful_connection = Some(timestamp);
                stats.active_peers += 1;
            }

            if let Some(err) = &error {
                *stats.error_patterns.entry(err.clone()).or_insert(0) += 1;
            }
        }

        // Detailed logging based on status
        match status {
            ConnectionStatus::Attempting => {
                info!("🔗 DEBUG CONNECTION [{} → {}]: ATTEMPTING via {} - {}",
                    self.node_id, target_peer, method, details);
            }
            ConnectionStatus::Connected => {
                info!("✅ DEBUG CONNECTION [{} → {}]: CONNECTED via {} in {}ms - {}",
                    self.node_id, target_peer, method,
                    latency_ms.unwrap_or(0), details);
            }
            ConnectionStatus::Failed => {
                error!("❌ DEBUG CONNECTION [{} → {}]: FAILED via {} - {} | Error: {}",
                    self.node_id, target_peer, method, details,
                    error.as_deref().unwrap_or("Unknown"));
            }
            ConnectionStatus::Timeout => {
                warn!("⏰ DEBUG CONNECTION [{} → {}]: TIMEOUT via {} after {}ms - {}",
                    self.node_id, target_peer, method,
                    latency_ms.unwrap_or(0), details);
            }
            ConnectionStatus::Rejected => {
                warn!("🚫 DEBUG CONNECTION [{} → {}]: REJECTED via {} - {} | Reason: {}",
                    self.node_id, target_peer, method, details,
                    error.as_deref().unwrap_or("Unknown"));
            }
        }
    }

    /// Generate detailed status report
    pub fn generate_debug_report(&self) -> String {
        let stats = self.network_stats.lock().unwrap();
        let discovery_attempts = self.discovery_attempts.lock().unwrap();
        let connection_attempts = self.connection_attempts.lock().unwrap();

        let mut report = String::new();

        report.push_str(&format!("\n🔧 === DISCOVERY & CONNECTION DEBUG REPORT ===\n"));
        report.push_str(&format!("Node ID: {}\n", self.node_id));
        report.push_str(&format!("Generated: {}\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));

        report.push_str(&format!("\n📊 NETWORK STATISTICS:\n"));
        report.push_str(&format!("• Total Discovery Attempts: {}\n", stats.total_discovery_attempts));
        report.push_str(&format!("• Successful Discoveries: {} ({:.1}%)\n",
            stats.successful_discoveries,
            if stats.total_discovery_attempts > 0 {
                (stats.successful_discoveries as f64 / stats.total_discovery_attempts as f64) * 100.0
            } else { 0.0 }));
        report.push_str(&format!("• Total Connection Attempts: {}\n", stats.total_connection_attempts));
        report.push_str(&format!("• Successful Connections: {} ({:.1}%)\n",
            stats.successful_connections,
            if stats.total_connection_attempts > 0 {
                (stats.successful_connections as f64 / stats.total_connection_attempts as f64) * 100.0
            } else { 0.0 }));
        report.push_str(&format!("• Active Peers: {}\n", stats.active_peers));

        report.push_str(&format!("\n🔍 DISCOVERY METHODS:\n"));
        for (method, count) in &stats.discovery_methods {
            report.push_str(&format!("• {}: {} attempts\n", method, count));
        }

        report.push_str(&format!("\n🔗 CONNECTION METHODS:\n"));
        for (method, count) in &stats.connection_methods {
            report.push_str(&format!("• {}: {} attempts\n", method, count));
        }

        if !stats.error_patterns.is_empty() {
            report.push_str(&format!("\n❌ TOP ERROR PATTERNS:\n"));
            let mut errors: Vec<_> = stats.error_patterns.iter().collect();
            errors.sort_by(|a, b| b.1.cmp(a.1));
            for (error, count) in errors.iter().take(5) {
                report.push_str(&format!("• [{}x] {}\n", count, error));
            }
        }

        report.push_str(&format!("\n🕒 RECENT DISCOVERY ATTEMPTS (Last 10):\n"));
        let mut recent_discoveries: Vec<_> = discovery_attempts.values().collect();
        recent_discoveries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        for attempt in recent_discoveries.iter().take(10) {
            let status_emoji = match attempt.status {
                DiscoveryStatus::PeersFound(count) if count > 0 => "✅",
                DiscoveryStatus::NopeersFound => "⚠️",
                DiscoveryStatus::Failed => "❌",
                _ => "🔍",
            };
            report.push_str(&format!("  {} {} via {}: {}\n",
                status_emoji,
                chrono::DateTime::from_timestamp(attempt.timestamp as i64, 0)
                    .map(|dt| dt.format("%H:%M:%S").to_string())
                    .unwrap_or("Unknown".to_string()),
                attempt.method,
                attempt.details));
        }

        report.push_str(&format!("\n🔗 RECENT CONNECTION ATTEMPTS (Last 10):\n"));
        let mut recent_connections: Vec<_> = connection_attempts.values().collect();
        recent_connections.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        for attempt in recent_connections.iter().take(10) {
            let status_emoji = match attempt.status {
                ConnectionStatus::Connected => "✅",
                ConnectionStatus::Failed => "❌",
                ConnectionStatus::Timeout => "⏰",
                ConnectionStatus::Rejected => "🚫",
                _ => "🔗",
            };
            report.push_str(&format!("  {} {} → {} via {}: {}\n",
                status_emoji,
                chrono::DateTime::from_timestamp(attempt.timestamp as i64, 0)
                    .map(|dt| dt.format("%H:%M:%S").to_string())
                    .unwrap_or("Unknown".to_string()),
                attempt.target_peer,
                attempt.method,
                attempt.connection_details));
        }

        report.push_str(&format!("\n🚨 DIAGNOSTIC RECOMMENDATIONS:\n"));

        if stats.total_discovery_attempts == 0 {
            report.push_str("• ❗ NO DISCOVERY ATTEMPTS DETECTED - Discovery services may not be running\n");
        } else if stats.successful_discoveries == 0 {
            report.push_str("• ❗ ZERO SUCCESSFUL DISCOVERIES - Check network connectivity and discovery services\n");
        }

        if stats.total_connection_attempts == 0 && stats.successful_discoveries > 0 {
            report.push_str("• ❗ PEERS DISCOVERED BUT NO CONNECTION ATTEMPTS - Connection logic may be broken\n");
        }

        if stats.total_connection_attempts > 0 && stats.successful_connections == 0 {
            report.push_str("• ❗ ALL CONNECTION ATTEMPTS FAILED - Check Tor service and network configuration\n");
        }

        if stats.discovery_methods.get("BEP-44").unwrap_or(&0) == &0 {
            report.push_str("• ⚠️  BEP-44 DHT discovery not active - Enable BitTorrent DHT\n");
        }

        if stats.discovery_methods.get("DNS-Phantom").unwrap_or(&0) == &0 {
            report.push_str("• ⚠️  DNS-Phantom discovery not active - Enable steganographic DNS discovery\n");
        }

        report.push_str(&format!("\n==============================================\n"));

        report
    }

    /// Start periodic debug reporting
    pub async fn start_periodic_reporting(self: Arc<Self>, interval_secs: u64) {
        let mut interval = interval(Duration::from_secs(interval_secs));

        loop {
            interval.tick().await;
            let report = self.generate_debug_report();
            info!("{}", report);
        }
    }

    /// Quick status check for immediate debugging
    pub fn quick_status(&self) -> String {
        let stats = self.network_stats.lock().unwrap();
        format!(
            "🔧 QUICK DEBUG: D:{}/{} C:{}/{} Peers:{}",
            stats.successful_discoveries, stats.total_discovery_attempts,
            stats.successful_connections, stats.total_connection_attempts,
            stats.active_peers
        )
    }
}

/// Integration trait for existing discovery systems
pub trait DebugDiscovery {
    fn set_debugger(&mut self, debugger: Arc<DiscoveryDebugger>);
    fn log_discovery(&self, method: &str, status: DiscoveryStatus, details: &str);
    fn log_connection(&self, target: &str, address: &str, method: &str, status: ConnectionStatus, details: &str);
}