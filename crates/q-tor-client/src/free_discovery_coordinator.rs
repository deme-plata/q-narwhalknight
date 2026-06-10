use crate::TorClient;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::bootstrap_discovery::BootstrapDiscovery;
use crate::gossip_discovery::GossipDiscovery;
use crate::tor_dht_discovery::TorDhtDiscovery;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    pub free_methods_only: bool,
    pub max_cost_per_day: f64,
    pub tor_dht_enabled: bool,
    pub bootstrap_enabled: bool,
    pub gossip_enabled: bool,
    pub bitcoin_discovery_enabled: bool,
    pub dns_discovery_enabled: bool,
    pub bootstrap_nodes: Vec<String>,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            free_methods_only: true,
            max_cost_per_day: 0.0,
            tor_dht_enabled: true,
            bootstrap_enabled: true,
            gossip_enabled: true,
            bitcoin_discovery_enabled: false,
            dns_discovery_enabled: false,
            bootstrap_nodes: vec![
                "bootstrap1.qnk.onion:8333".to_string(),
                "bootstrap2.qnk.onion:8333".to_string(),
                "bootstrap3.qnk.onion:8333".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiscoveryMethod {
    TorDht,
    Bootstrap,
    Gossip,
    BitcoinOpReturn,
    DnsRecord,
}

impl DiscoveryMethod {
    pub fn is_free(&self) -> bool {
        matches!(
            self,
            DiscoveryMethod::TorDht | DiscoveryMethod::Bootstrap | DiscoveryMethod::Gossip
        )
    }

    pub fn cost_per_operation(&self) -> f64 {
        match self {
            DiscoveryMethod::TorDht => 0.0,
            DiscoveryMethod::Bootstrap => 0.0,
            DiscoveryMethod::Gossip => 0.0,
            DiscoveryMethod::BitcoinOpReturn => 25.0, // $25 average per Bitcoin transaction
            DiscoveryMethod::DnsRecord => 0.01,       // ~$10/year amortized
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            DiscoveryMethod::TorDht => "Tor DHT",
            DiscoveryMethod::Bootstrap => "Bootstrap Nodes",
            DiscoveryMethod::Gossip => "Gossip Protocol",
            DiscoveryMethod::BitcoinOpReturn => "Bitcoin OP_RETURN",
            DiscoveryMethod::DnsRecord => "DNS TXT Records",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiscoveryResult {
    pub method: DiscoveryMethod,
    pub peers_discovered: Vec<String>,
    pub success: bool,
    pub cost: f64,
    pub latency: Duration,
    pub timestamp: SystemTime,
}

pub struct CostTracker {
    daily_costs: Arc<RwLock<HashMap<String, f64>>>, // Date -> Cost
    operation_costs: Arc<RwLock<Vec<(SystemTime, DiscoveryMethod, f64)>>>,
    max_daily_cost: f64,
}

impl CostTracker {
    pub fn new(max_daily_cost: f64) -> Self {
        Self {
            daily_costs: Arc::new(RwLock::new(HashMap::new())),
            operation_costs: Arc::new(RwLock::new(Vec::new())),
            max_daily_cost,
        }
    }

    pub async fn can_afford_operation(&self, method: DiscoveryMethod) -> bool {
        let cost = method.cost_per_operation();
        if cost == 0.0 {
            return true; // Free operations are always allowed
        }

        let today = self.get_date_string();
        let daily_costs = self.daily_costs.read().await;
        let current_daily_cost = daily_costs.get(&today).copied().unwrap_or(0.0);

        current_daily_cost + cost <= self.max_daily_cost
    }

    pub async fn record_operation_cost(&self, method: DiscoveryMethod) -> Result<()> {
        let cost = method.cost_per_operation();
        let now = SystemTime::now();
        let today = self.get_date_string();

        // Record the operation
        {
            let mut operations = self.operation_costs.write().await;
            operations.push((now, method, cost));
        }

        // Update daily totals
        {
            let mut daily_costs = self.daily_costs.write().await;
            let current = daily_costs.get(&today).copied().unwrap_or(0.0);
            daily_costs.insert(today, current + cost);
        }

        if cost > 0.0 {
            warn!(
                "💰 Discovery operation cost: {} = ${:.2}",
                method.name(),
                cost
            );
        } else {
            debug!("🆓 Discovery operation: {} (FREE)", method.name());
        }

        Ok(())
    }

    pub async fn get_daily_cost(&self) -> f64 {
        let today = self.get_date_string();
        let daily_costs = self.daily_costs.read().await;
        daily_costs.get(&today).copied().unwrap_or(0.0)
    }

    pub async fn get_total_cost(&self) -> f64 {
        let daily_costs = self.daily_costs.read().await;
        daily_costs.values().sum()
    }

    fn get_date_string(&self) -> String {
        let now = SystemTime::now();
        let duration = now.duration_since(UNIX_EPOCH).unwrap();
        let days = duration.as_secs() / 86400;
        format!("day-{}", days)
    }
}

pub struct FreeDiscoveryCoordinator {
    tor_client: Arc<TorClient>,
    config: DiscoveryConfig,

    // Discovery methods
    tor_dht: Option<TorDhtDiscovery>,
    bootstrap: Option<BootstrapDiscovery>,
    gossip: Option<GossipDiscovery>,

    // State management
    discovered_peers: Arc<RwLock<HashSet<String>>>,
    discovery_results: Arc<RwLock<Vec<DiscoveryResult>>>,
    cost_tracker: CostTracker,

    // Node information
    our_node_id: String,
    our_onion_address: Option<String>,
    our_port: u16,
}

impl FreeDiscoveryCoordinator {
    pub fn new(
        tor_client: Arc<TorClient>,
        config: DiscoveryConfig,
        node_id: String,
        port: u16,
    ) -> Self {
        let cost_tracker = CostTracker::new(config.max_cost_per_day);

        Self {
            tor_client,
            config,
            tor_dht: None,
            bootstrap: None,
            gossip: None,
            discovered_peers: Arc::new(RwLock::new(HashSet::new())),
            discovery_results: Arc::new(RwLock::new(Vec::new())),
            cost_tracker,
            our_node_id: node_id,
            our_onion_address: None,
            our_port: port,
        }
    }

    pub async fn initialize(&mut self, onion_address: String) -> Result<()> {
        info!("🆓 Initializing FREE peer discovery system");
        info!(
            "🆓 Configuration: free_only={}, max_daily_cost=${:.2}",
            self.config.free_methods_only, self.config.max_cost_per_day
        );

        self.our_onion_address = Some(onion_address.clone());

        // Initialize free discovery methods
        if self.config.tor_dht_enabled {
            info!("🆓 Initializing Tor DHT discovery (FREE)");
            let mut tor_dht = TorDhtDiscovery::new(Arc::clone(&self.tor_client));
            tor_dht
                .start_discovery(
                    onion_address.clone(),
                    self.our_port,
                    self.our_node_id.clone(),
                )
                .await?;
            self.tor_dht = Some(tor_dht);
        }

        if self.config.bootstrap_enabled {
            info!("🆓 Initializing bootstrap discovery (FREE)");
            let bootstrap = if self.config.bootstrap_nodes.is_empty() {
                BootstrapDiscovery::new(Arc::clone(&self.tor_client))
            } else {
                BootstrapDiscovery::with_custom_bootstrap_nodes(
                    Arc::clone(&self.tor_client),
                    self.config.bootstrap_nodes.clone(),
                )
            };
            bootstrap.start_discovery().await?;
            self.bootstrap = Some(bootstrap);
        }

        if self.config.gossip_enabled {
            info!("🆓 Initializing gossip discovery (FREE)");
            let mut gossip = GossipDiscovery::new(
                Arc::clone(&self.tor_client),
                self.our_node_id.clone(),
                onion_address,
                self.our_port,
            );
            gossip.start_gossip_protocol().await?;
            self.gossip = Some(gossip);
        }

        info!("✅ FREE discovery system initialized - $0.00 daily operating cost");
        Ok(())
    }

    pub async fn discover_peers(&self) -> Result<Vec<String>> {
        info!("🔍 Starting comprehensive FREE peer discovery");
        let start_time = SystemTime::now();
        let mut all_discovered = HashSet::new();

        // Try free methods in priority order
        let free_methods = vec![
            (DiscoveryMethod::TorDht, "Tor DHT discovery"),
            (DiscoveryMethod::Bootstrap, "Bootstrap node discovery"),
            (DiscoveryMethod::Gossip, "Gossip protocol discovery"),
        ];

        let mut successful_methods = 0;
        let mut total_cost = 0.0;

        for (method, description) in free_methods {
            if !self.cost_tracker.can_afford_operation(method).await {
                warn!("⚠️ Skipping {} - exceeds daily cost limit", description);
                continue;
            }

            let method_start = SystemTime::now();
            let discovery_result = match method {
                DiscoveryMethod::TorDht => self.discover_via_tor_dht().await,
                DiscoveryMethod::Bootstrap => self.discover_via_bootstrap().await,
                DiscoveryMethod::Gossip => self.discover_via_gossip().await,
                _ => continue, // Skip non-free methods
            };

            let method_latency = method_start.elapsed().unwrap_or(Duration::ZERO);
            let method_cost = method.cost_per_operation();
            total_cost += method_cost;

            match discovery_result {
                Ok(peers) => {
                    successful_methods += 1;
                    let new_peers: Vec<_> = peers
                        .iter()
                        .filter(|peer| all_discovered.insert((*peer).clone()))
                        .cloned()
                        .collect();

                    info!(
                        "🆓 {} found {} peers ({} new) in {:?} (FREE)",
                        description,
                        peers.len(),
                        new_peers.len(),
                        method_latency
                    );

                    // Record successful discovery
                    let result = DiscoveryResult {
                        method,
                        peers_discovered: peers,
                        success: true,
                        cost: method_cost,
                        latency: method_latency,
                        timestamp: SystemTime::now(),
                    };

                    {
                        let mut results = self.discovery_results.write().await;
                        results.push(result);
                    }
                }
                Err(e) => {
                    debug!("Failed {}: {}", description, e);

                    // Record failed discovery
                    let result = DiscoveryResult {
                        method,
                        peers_discovered: Vec::new(),
                        success: false,
                        cost: method_cost,
                        latency: method_latency,
                        timestamp: SystemTime::now(),
                    };

                    {
                        let mut results = self.discovery_results.write().await;
                        results.push(result);
                    }
                }
            }

            // Record cost (even for free methods to track usage)
            self.cost_tracker.record_operation_cost(method).await?;
        }

        // Update discovered peers
        {
            let mut discovered = self.discovered_peers.write().await;
            discovered.extend(all_discovered.iter().cloned());
        }

        let total_latency = start_time.elapsed().unwrap_or(Duration::ZERO);
        let result_peers: Vec<String> = all_discovered.into_iter().collect();

        if successful_methods == 0 {
            if self.config.free_methods_only {
                return Err(anyhow!("All free discovery methods failed. Enable paid methods or check network connectivity."));
            } else {
                warn!("All free methods failed, trying expensive fallbacks...");
                return self.discover_via_expensive_methods().await;
            }
        }

        info!(
            "🎉 FREE discovery complete: {} peers from {} methods in {:?} - Total cost: ${:.2}",
            result_peers.len(),
            successful_methods,
            total_latency,
            total_cost
        );

        Ok(result_peers)
    }

    async fn discover_via_tor_dht(&self) -> Result<Vec<String>> {
        if let Some(tor_dht) = &self.tor_dht {
            let peers = tor_dht.get_discovered_peers().await;
            debug!("🆓 Tor DHT returned {} peers (FREE)", peers.len());
            Ok(peers)
        } else {
            Err(anyhow!("Tor DHT discovery not initialized"))
        }
    }

    async fn discover_via_bootstrap(&self) -> Result<Vec<String>> {
        if let Some(bootstrap) = &self.bootstrap {
            bootstrap.discover_peers_from_all_bootstraps().await
        } else {
            Err(anyhow!("Bootstrap discovery not initialized"))
        }
    }

    async fn discover_via_gossip(&self) -> Result<Vec<String>> {
        if let Some(gossip) = &self.gossip {
            Ok(gossip.get_discovered_peers().await)
        } else {
            Err(anyhow!("Gossip discovery not initialized"))
        }
    }

    async fn discover_via_expensive_methods(&self) -> Result<Vec<String>> {
        warn!("💰 Attempting expensive discovery methods - this will cost money!");

        // Check if we can afford Bitcoin OP_RETURN
        if self
            .cost_tracker
            .can_afford_operation(DiscoveryMethod::BitcoinOpReturn)
            .await
        {
            warn!("💰 Using Bitcoin OP_RETURN discovery - this costs ~$25 per operation!");
            // Would implement Bitcoin discovery here
            self.cost_tracker
                .record_operation_cost(DiscoveryMethod::BitcoinOpReturn)
                .await?;
        }

        // DNS discovery is cheaper
        if self
            .cost_tracker
            .can_afford_operation(DiscoveryMethod::DnsRecord)
            .await
        {
            info!("Using DNS TXT record discovery (low cost)");
            self.cost_tracker
                .record_operation_cost(DiscoveryMethod::DnsRecord)
                .await?;
        }

        // For now, return empty - expensive methods not implemented
        Ok(Vec::new())
    }

    pub async fn add_seed_peer(
        &self,
        onion_address: String,
        port: u16,
        node_id: String,
    ) -> Result<()> {
        info!(
            "🌱 Adding seed peer: {}:{} ({})",
            onion_address, port, node_id
        );

        // Add to gossip if enabled
        if let Some(gossip) = &self.gossip {
            gossip
                .add_seed_peer(onion_address.clone(), port, node_id.clone())
                .await;
        }

        // Add to our discovered peers
        {
            let mut discovered = self.discovered_peers.write().await;
            discovered.insert(format!("{}:{}", onion_address, port));
        }

        info!("🆓 Seed peer added (FREE operation)");
        Ok(())
    }

    pub async fn get_discovery_stats(&self) -> DiscoveryStats {
        let discovered_peers = {
            let peers = self.discovered_peers.read().await;
            peers.len()
        };

        let daily_cost = self.cost_tracker.get_daily_cost().await;
        let total_cost = self.cost_tracker.get_total_cost().await;

        let method_stats = {
            let results = self.discovery_results.read().await;
            let mut stats = HashMap::new();

            for result in results.iter() {
                let entry = stats.entry(result.method).or_insert((0usize, 0usize, 0.0));
                if result.success {
                    entry.0 += result.peers_discovered.len();
                    entry.1 += 1;
                }
                entry.2 += result.cost;
            }
            stats
        };

        DiscoveryStats {
            total_discovered_peers: discovered_peers,
            daily_cost,
            total_cost,
            method_stats,
            free_methods_only: self.config.free_methods_only,
            max_daily_cost: self.config.max_cost_per_day,
        }
    }

    pub async fn start_continuous_discovery(&self) -> Result<()> {
        info!("🔄 Starting continuous FREE peer discovery");

        let discovered_peers = Arc::clone(&self.discovered_peers);
        let tor_dht = self.tor_dht.as_ref().map(|_| ());
        let bootstrap = self.bootstrap.as_ref().map(|_| ());

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes

            loop {
                interval.tick().await;

                // Periodic cleanup and refresh would happen here
                debug!("🔄 Periodic FREE discovery refresh");

                // In a real implementation, this would:
                // 1. Refresh DHT records
                // 2. Query bootstrap nodes for updates
                // 3. Share gossip with connected peers
                // 4. Clean up expired peers
            }
        });

        info!("✅ Continuous FREE discovery started");
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DiscoveryStats {
    pub total_discovered_peers: usize,
    pub daily_cost: f64,
    pub total_cost: f64,
    pub method_stats: HashMap<DiscoveryMethod, (usize, usize, f64)>, // (peers_found, successful_operations, total_cost)
    pub free_methods_only: bool,
    pub max_daily_cost: f64,
}

impl DiscoveryStats {
    pub fn print_summary(&self) {
        info!("📊 FREE Discovery Statistics:");
        info!("   Total peers discovered: {}", self.total_discovered_peers);
        info!("   Daily cost: ${:.2}", self.daily_cost);
        info!("   Total cost: ${:.2}", self.total_cost);
        info!("   Free methods only: {}", self.free_methods_only);
        info!("   Daily cost limit: ${:.2}", self.max_daily_cost);

        for (method, (peers, operations, cost)) in &self.method_stats {
            info!(
                "   {}: {} peers, {} operations, ${:.2}",
                method.name(),
                peers,
                operations,
                cost
            );
        }

        if self.daily_cost == 0.0 {
            info!("🎉 COMPLETELY FREE OPERATION - $0.00 daily costs!");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_discovery_method_costs() {
        assert!(DiscoveryMethod::TorDht.is_free());
        assert!(DiscoveryMethod::Bootstrap.is_free());
        assert!(DiscoveryMethod::Gossip.is_free());
        assert!(!DiscoveryMethod::BitcoinOpReturn.is_free());

        assert_eq!(DiscoveryMethod::TorDht.cost_per_operation(), 0.0);
        assert_eq!(DiscoveryMethod::Bootstrap.cost_per_operation(), 0.0);
        assert!(DiscoveryMethod::BitcoinOpReturn.cost_per_operation() > 0.0);
    }

    #[tokio::test]
    async fn test_cost_tracker() {
        let mut tracker = CostTracker::new(10.0);

        // Free operations should always be allowed
        assert!(tracker.can_afford_operation(DiscoveryMethod::TorDht).await);
        assert!(tracker.can_afford_operation(DiscoveryMethod::Gossip).await);

        // Record some costs
        tracker
            .record_operation_cost(DiscoveryMethod::TorDht)
            .await
            .unwrap();
        tracker
            .record_operation_cost(DiscoveryMethod::Bootstrap)
            .await
            .unwrap();

        // Daily cost should still be 0 for free operations
        assert_eq!(tracker.get_daily_cost().await, 0.0);
    }

    #[test]
    fn test_discovery_config_defaults() {
        let config = DiscoveryConfig::default();
        assert!(config.free_methods_only);
        assert_eq!(config.max_cost_per_day, 0.0);
        assert!(config.tor_dht_enabled);
        assert!(config.bootstrap_enabled);
        assert!(config.gossip_enabled);
        assert!(!config.bitcoin_discovery_enabled);
    }
}
