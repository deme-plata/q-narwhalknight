/// Discovery Metrics for Q-NarwhalKnight Zero-Knowledge Discovery
/// Provides structured monitoring for mDNS, Kademlia, and Gossipsub discovery

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Default instant for serde deserialization
fn default_instant() -> Instant {
    Instant::now()
}

/// Comprehensive discovery metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryMetrics {
    /// mDNS local discovery metrics
    pub mdns: MdnsMetrics,
    /// Kademlia global DHT metrics
    pub kademlia: KademliaMetrics,
    /// Gossipsub peer amplification metrics
    pub gossipsub: GossipsubMetrics,
    /// Overall discovery performance
    pub overall: OverallMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdnsMetrics {
    /// Number of peers discovered via mDNS
    pub peers_discovered: u64,
    /// Average discovery latency (milliseconds)
    pub avg_discovery_latency_ms: f64,
    /// Last discovery timestamp (as seconds since start)
    #[serde(skip)]
    pub last_discovery: Option<Instant>,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KademliaMetrics {
    /// Number of bootstrap nodes connected
    pub bootstrap_nodes_connected: u64,
    /// Number of Q-NarwhalKnight peers found via DHT
    pub qnarwhal_peers_found: u64,
    /// Average bootstrap latency (milliseconds)
    pub avg_bootstrap_latency_ms: f64,
    /// DHT query success rate
    pub query_success_rate: f64,
    /// Last successful bootstrap
    #[serde(skip)]
    pub last_bootstrap: Option<Instant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipsubMetrics {
    /// Number of peers connected via gossipsub
    pub connected_peers: u64,
    /// Messages published
    pub messages_published: u64,
    /// Messages received
    pub messages_received: u64,
    /// Average message propagation time (milliseconds)
    pub avg_propagation_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallMetrics {
    /// Total unique peers discovered
    pub total_peers_discovered: u64,
    /// Average time to first peer (milliseconds)
    pub avg_time_to_first_peer_ms: f64,
    /// Combined success rate across all mechanisms
    pub combined_success_rate: f64,
    /// Discovery start time
    #[serde(skip)]
    #[serde(default = "default_instant")]
    pub discovery_start: Instant,
    /// Uptime in seconds
    pub uptime_seconds: u64,
}

/// Thread-safe metrics collector
pub struct DiscoveryMetricsCollector {
    metrics: Arc<RwLock<DiscoveryMetrics>>,
    start_time: Instant,
}

impl DiscoveryMetricsCollector {
    pub fn new() -> Self {
        let start_time = Instant::now();

        Self {
            metrics: Arc::new(RwLock::new(DiscoveryMetrics {
                mdns: MdnsMetrics {
                    peers_discovered: 0,
                    avg_discovery_latency_ms: 0.0,
                    last_discovery: None,
                    success_rate: 0.0,
                },
                kademlia: KademliaMetrics {
                    bootstrap_nodes_connected: 0,
                    qnarwhal_peers_found: 0,
                    avg_bootstrap_latency_ms: 0.0,
                    query_success_rate: 0.0,
                    last_bootstrap: None,
                },
                gossipsub: GossipsubMetrics {
                    connected_peers: 0,
                    messages_published: 0,
                    messages_received: 0,
                    avg_propagation_ms: 0.0,
                },
                overall: OverallMetrics {
                    total_peers_discovered: 0,
                    avg_time_to_first_peer_ms: 0.0,
                    combined_success_rate: 0.0,
                    discovery_start: start_time,
                    uptime_seconds: 0,
                },
            })),
            start_time,
        }
    }

    /// Record mDNS peer discovery
    pub async fn record_mdns_discovery(&self, latency: Duration) {
        let mut metrics = self.metrics.write().await;
        metrics.mdns.peers_discovered += 1;
        metrics.mdns.last_discovery = Some(Instant::now());

        // Update average latency
        let new_latency_ms = latency.as_millis() as f64;
        metrics.mdns.avg_discovery_latency_ms =
            (metrics.mdns.avg_discovery_latency_ms * (metrics.mdns.peers_discovered - 1) as f64 + new_latency_ms)
            / metrics.mdns.peers_discovered as f64;

        // Update overall metrics
        metrics.overall.total_peers_discovered += 1;
    }

    /// Record Kademlia bootstrap success
    pub async fn record_kademlia_bootstrap(&self, latency: Duration) {
        let mut metrics = self.metrics.write().await;
        metrics.kademlia.bootstrap_nodes_connected += 1;
        metrics.kademlia.last_bootstrap = Some(Instant::now());

        // Update average bootstrap latency
        let new_latency_ms = latency.as_millis() as f64;
        metrics.kademlia.avg_bootstrap_latency_ms =
            (metrics.kademlia.avg_bootstrap_latency_ms * (metrics.kademlia.bootstrap_nodes_connected - 1) as f64 + new_latency_ms)
            / metrics.kademlia.bootstrap_nodes_connected as f64;
    }

    /// Record Q-NarwhalKnight peer found via DHT
    pub async fn record_qnarwhal_peer_found(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.kademlia.qnarwhal_peers_found += 1;
        metrics.overall.total_peers_discovered += 1;
    }

    /// Record gossipsub message
    pub async fn record_gossipsub_message(&self, is_outbound: bool, propagation_time: Option<Duration>) {
        let mut metrics = self.metrics.write().await;

        if is_outbound {
            metrics.gossipsub.messages_published += 1;
        } else {
            metrics.gossipsub.messages_received += 1;

            if let Some(prop_time) = propagation_time {
                let prop_ms = prop_time.as_millis() as f64;
                let total_messages = metrics.gossipsub.messages_received;
                metrics.gossipsub.avg_propagation_ms =
                    (metrics.gossipsub.avg_propagation_ms * (total_messages - 1) as f64 + prop_ms)
                    / total_messages as f64;
            }
        }
    }

    /// Get current metrics (for Prometheus endpoint)
    pub async fn get_metrics(&self) -> DiscoveryMetrics {
        let mut metrics = self.metrics.read().await.clone();

        // Update uptime
        metrics.overall.uptime_seconds = self.start_time.elapsed().as_secs();

        // Calculate combined success rate (simplified)
        let mdns_weight = 0.4;
        let kad_weight = 0.4;
        let gossip_weight = 0.2;

        metrics.overall.combined_success_rate =
            metrics.mdns.success_rate * mdns_weight +
            metrics.kademlia.query_success_rate * kad_weight +
            if metrics.gossipsub.connected_peers > 0 { 1.0 } else { 0.0 } * gossip_weight;

        metrics
    }

    /// Generate Prometheus metrics format
    pub async fn to_prometheus_format(&self) -> String {
        let metrics = self.get_metrics().await;

        format!(
            r#"# HELP qnarwhal_mdns_peers_discovered_total Number of peers discovered via mDNS
# TYPE qnarwhal_mdns_peers_discovered_total counter
qnarwhal_mdns_peers_discovered_total {}

# HELP qnarwhal_mdns_discovery_latency_ms Average mDNS discovery latency in milliseconds
# TYPE qnarwhal_mdns_discovery_latency_ms gauge
qnarwhal_mdns_discovery_latency_ms {:.2}

# HELP qnarwhal_kademlia_bootstrap_nodes_total Number of Kademlia bootstrap nodes connected
# TYPE qnarwhal_kademlia_bootstrap_nodes_total counter
qnarwhal_kademlia_bootstrap_nodes_total {}

# HELP qnarwhal_kademlia_peers_found_total Number of Q-NarwhalKnight peers found via DHT
# TYPE qnarwhal_kademlia_peers_found_total counter
qnarwhal_kademlia_peers_found_total {}

# HELP qnarwhal_kademlia_bootstrap_latency_ms Average Kademlia bootstrap latency in milliseconds
# TYPE qnarwhal_kademlia_bootstrap_latency_ms gauge
qnarwhal_kademlia_bootstrap_latency_ms {:.2}

# HELP qnarwhal_gossipsub_connected_peers Number of peers connected via gossipsub
# TYPE qnarwhal_gossipsub_connected_peers gauge
qnarwhal_gossipsub_connected_peers {}

# HELP qnarwhal_gossipsub_messages_published_total Number of gossipsub messages published
# TYPE qnarwhal_gossipsub_messages_published_total counter
qnarwhal_gossipsub_messages_published_total {}

# HELP qnarwhal_gossipsub_messages_received_total Number of gossipsub messages received
# TYPE qnarwhal_gossipsub_messages_received_total counter
qnarwhal_gossipsub_messages_received_total {}

# HELP qnarwhal_discovery_total_peers_total Total unique peers discovered across all mechanisms
# TYPE qnarwhal_discovery_total_peers_total counter
qnarwhal_discovery_total_peers_total {}

# HELP qnarwhal_discovery_success_rate Combined discovery success rate (0.0-1.0)
# TYPE qnarwhal_discovery_success_rate gauge
qnarwhal_discovery_success_rate {:.2}

# HELP qnarwhal_discovery_uptime_seconds Discovery system uptime in seconds
# TYPE qnarwhal_discovery_uptime_seconds counter
qnarwhal_discovery_uptime_seconds {}
"#,
            metrics.mdns.peers_discovered,
            metrics.mdns.avg_discovery_latency_ms,
            metrics.kademlia.bootstrap_nodes_connected,
            metrics.kademlia.qnarwhal_peers_found,
            metrics.kademlia.avg_bootstrap_latency_ms,
            metrics.gossipsub.connected_peers,
            metrics.gossipsub.messages_published,
            metrics.gossipsub.messages_received,
            metrics.overall.total_peers_discovered,
            metrics.overall.combined_success_rate,
            metrics.overall.uptime_seconds
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_collection() {
        let collector = DiscoveryMetricsCollector::new();

        // Record some discoveries
        collector.record_mdns_discovery(Duration::from_millis(800)).await;
        collector.record_kademlia_bootstrap(Duration::from_secs(12)).await;
        collector.record_qnarwhal_peer_found().await;

        let metrics = collector.get_metrics().await;
        assert_eq!(metrics.mdns.peers_discovered, 1);
        assert_eq!(metrics.kademlia.qnarwhal_peers_found, 1);
        assert_eq!(metrics.overall.total_peers_discovered, 2);
    }

    #[tokio::test]
    async fn test_prometheus_format() {
        let collector = DiscoveryMetricsCollector::new();
        collector.record_mdns_discovery(Duration::from_millis(500)).await;

        let prometheus = collector.to_prometheus_format().await;
        assert!(prometheus.contains("qnarwhal_mdns_peers_discovered_total 1"));
        assert!(prometheus.contains("qnarwhal_mdns_discovery_latency_ms 500.00"));
    }
}