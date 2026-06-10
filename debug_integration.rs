use std::sync::Arc;
use tracing::{info, warn, error};
use tokio::time::{interval, Duration};
use q_network::discovery_debug::{DiscoveryDebugger, DiscoveryStatus, ConnectionStatus};

/// Comprehensive debugging integration for Q-NarwhalKnight
#[derive(Clone)]
pub struct DebugIntegration {
    debugger: Arc<DiscoveryDebugger>,
    node_id: String,
}

impl DebugIntegration {
    pub fn new(node_id: String) -> Self {
        let debugger = Arc::new(DiscoveryDebugger::new(node_id.clone()));

        Self {
            debugger,
            node_id,
        }
    }

    pub fn get_debugger(&self) -> Arc<DiscoveryDebugger> {
        self.debugger.clone()
    }

    /// Start comprehensive debugging monitoring
    pub async fn start_debugging_monitor(&self) {
        info!("🔧 DEBUG: Starting comprehensive debugging monitor for node: {}", self.node_id);

        let debugger = self.debugger.clone();

        // Spawn periodic status reporting task
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Report every minute

            loop {
                interval.tick().await;

                // Generate and log comprehensive debug report
                let report = debugger.generate_debug_report();
                info!("{}", report);

                // Quick status check
                let quick_status = debugger.quick_status();
                info!("🔧 QUICK DEBUG STATUS: {}", quick_status);
            }
        });
    }

    /// Log discovery attempts from various discovery systems
    pub fn log_bep44_discovery(&self, peers_found: u32, details: &str) {
        self.debugger.log_discovery_attempt(
            "BEP-44",
            None,
            None,
            if peers_found > 0 {
                DiscoveryStatus::PeersFound(peers_found)
            } else {
                DiscoveryStatus::NopeersFound
            },
            details,
            None,
        );
    }

    pub fn log_dns_phantom_discovery(&self, peers_found: u32, details: &str) {
        self.debugger.log_discovery_attempt(
            "DNS-Phantom",
            None,
            None,
            if peers_found > 0 {
                DiscoveryStatus::PeersFound(peers_found)
            } else {
                DiscoveryStatus::NopeersFound
            },
            details,
            None,
        );
    }

    pub fn log_bitcoin_bridge_discovery(&self, peers_found: u32, details: &str, error: Option<String>) {
        self.debugger.log_discovery_attempt(
            "Bitcoin-Bridge",
            None,
            None,
            if error.is_some() {
                DiscoveryStatus::Failed
            } else if peers_found > 0 {
                DiscoveryStatus::PeersFound(peers_found)
            } else {
                DiscoveryStatus::NopeersFound
            },
            details,
            error,
        );
    }

    pub fn log_production_peer_discovery(&self, peers_found: u32, details: &str) {
        self.debugger.log_discovery_attempt(
            "Production-Discovery",
            None,
            None,
            if peers_found > 0 {
                DiscoveryStatus::PeersFound(peers_found)
            } else {
                DiscoveryStatus::NopeersFound
            },
            details,
            None,
        );
    }

    /// Log peer connection events
    pub fn log_peer_discovered(&self, method: &str, peer_id: &str, onion_address: Option<String>) {
        self.debugger.log_discovery_attempt(
            method,
            Some(peer_id.to_string()),
            onion_address,
            DiscoveryStatus::PeersFound(1),
            &format!("Peer {} discovered", peer_id),
            None,
        );
    }

    /// Log Tor-related connection attempts
    pub fn log_tor_connection(&self, target: &str, onion_address: &str, success: bool, error: Option<String>) {
        self.debugger.log_connection_attempt(
            target,
            onion_address,
            "Tor",
            if success { ConnectionStatus::Connected } else { ConnectionStatus::Failed },
            None,
            error,
            "Tor onion service connection"
        );
    }

    /// Force generate debug report for immediate analysis
    pub fn generate_immediate_report(&self) -> String {
        self.debugger.generate_debug_report()
    }

    /// Get quick status for real-time monitoring
    pub fn get_quick_status(&self) -> String {
        self.debugger.quick_status()
    }
}