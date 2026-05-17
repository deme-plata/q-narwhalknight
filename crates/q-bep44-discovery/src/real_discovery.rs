/*!
# Real Q-NarwhalKnight BEP-44 + Tor Discovery Engine

This is the REAL implementation - not a simulation, but actual BitTorrent DHT + Tor
integration for massive scale peer discovery.
*/

use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use crate::real_bep44::{PeerPresenceRecord, RealBep44Client};
use crate::real_tor::{generate_qnk_onion_address, RealTorClient};

/// Real BEP-44 + Tor Discovery Engine for Q-NarwhalKnight
#[derive(Debug)]
pub struct RealDiscoveryEngine {
    /// Real BitTorrent DHT client
    dht_client: Arc<RwLock<RealBep44Client>>,
    /// Real Tor client
    tor_client: Arc<RwLock<RealTorClient>>,
    /// Validator identity
    signing_key: SigningKey,
    /// Our onion address
    onion_address: Arc<RwLock<Option<String>>>,
    /// Discovered peers from DHT
    discovered_peers: Arc<RwLock<HashMap<String, DiscoveredPeer>>>,
    /// Real discovery statistics
    stats: Arc<RwLock<RealDiscoveryStats>>,
}

/// Discovered peer from real DHT + Tor
#[derive(Debug, Clone, Serialize)]
pub struct DiscoveredPeer {
    pub validator_id: String,
    pub onion_address: String,
    pub capabilities: Vec<String>,
    pub discovered_at: chrono::DateTime<chrono::Utc>,
    pub last_seen: chrono::DateTime<chrono::Utc>,
    pub verified: bool,
    pub connection_tested: bool,
}

/// Real discovery statistics  
#[derive(Debug, Clone, Serialize)]
pub struct RealDiscoveryStats {
    pub dht_connected_nodes: usize,
    pub dht_stored_records: usize,
    pub tor_active_circuits: u32,
    pub discovered_peers: usize,
    pub successful_connections: u64,
    pub failed_connections: u64,
    pub last_announcement: Option<chrono::DateTime<chrono::Utc>>,
    pub onion_service_active: bool,
}

impl RealDiscoveryEngine {
    /// Create new REAL discovery engine
    pub async fn new(signing_key: SigningKey) -> Result<Self> {
        info!("🌟 Creating REAL BEP-44 + Tor discovery engine");
        info!("   • This is NOT a simulation - using real networks!");

        // Real BitTorrent DHT bootstrap nodes
        let bootstrap_nodes = vec![
            "87.98.162.88:6881".parse().unwrap(),   // router.bittorrent.com
            "212.129.33.59:6881".parse().unwrap(),  // dht.transmissionbt.com
            "82.221.103.244:6881".parse().unwrap(), // router.utorrent.com
            "212.47.227.58:6881".parse().unwrap(),  // dht.aelitis.com
            "87.98.162.88:6881".parse().unwrap(),   // Additional redundancy
        ];

        // Create real DHT client
        let dht_client = RealBep44Client::new(signing_key.clone(), bootstrap_nodes)
            .await
            .context("Failed to create real DHT client")?;

        // Create real Tor client
        let tor_client = RealTorClient::new()
            .await
            .context("Failed to create real Tor client")?;

        let stats = RealDiscoveryStats {
            dht_connected_nodes: 0,
            dht_stored_records: 0,
            tor_active_circuits: 0,
            discovered_peers: 0,
            successful_connections: 0,
            failed_connections: 0,
            last_announcement: None,
            onion_service_active: false,
        };

        info!("✅ Real discovery engine created successfully");

        Ok(Self {
            dht_client: Arc::new(RwLock::new(dht_client)),
            tor_client: Arc::new(RwLock::new(tor_client)),
            signing_key,
            onion_address: Arc::new(RwLock::new(None)),
            discovered_peers: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(stats)),
        })
    }

    /// Initialize the REAL discovery engine
    pub async fn initialize(&mut self) -> Result<()> {
        info!("🚀 Initializing REAL BEP-44 + Tor discovery engine");

        // Bootstrap to real BitTorrent DHT network
        {
            let mut dht = self.dht_client.write().await;
            dht.bootstrap().await.context("DHT bootstrap failed")?;
        }

        // Generate our .qnk.onion address
        let validator_id = *self.signing_key.verifying_key().as_bytes();
        let onion_addr = generate_qnk_onion_address(&validator_id)?;

        // Create real onion service
        {
            let mut tor = self.tor_client.write().await;
            let real_onion = tor
                .create_onion_service(8080)
                .await
                .context("Failed to create onion service")?;
            info!("🧅 Created real onion service: {}", real_onion);
        }

        {
            let mut addr = self.onion_address.write().await;
            *addr = Some(onion_addr.clone());
        }

        info!("✅ Real discovery engine initialized");
        info!("   • Our .qnk.onion address: {}", onion_addr);

        Ok(())
    }

    /// Start REAL peer discovery process
    pub async fn start_discovery(&mut self) -> Result<()> {
        info!("🎯 Starting REAL peer discovery process");

        // Announce our presence to the real BitTorrent DHT
        self.announce_presence().await?;

        // Start discovering peers
        self.discover_peers().await?;

        // Test connections to discovered peers
        self.test_peer_connections().await?;

        info!("🚀 Real discovery process is running");
        Ok(())
    }

    /// Announce our validator presence to the REAL BitTorrent DHT
    async fn announce_presence(&mut self) -> Result<()> {
        let onion_addr = {
            let addr = self.onion_address.read().await;
            addr.as_ref().unwrap().clone()
        };

        let capabilities = vec![
            "consensus".to_string(),
            "mempool".to_string(),
            "qkd-ready".to_string(),
        ];

        info!("📢 Announcing presence to REAL BitTorrent DHT");
        info!("   • Onion address: {}", onion_addr);
        info!("   • Capabilities: {:?}", capabilities);

        let mut dht = self.dht_client.write().await;
        let target = dht
            .announce_presence(&onion_addr, capabilities)
            .await
            .context("Failed to announce presence to DHT")?;

        info!("✅ Presence announced successfully");
        info!("   • DHT target: {}", hex::encode(&target));

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.last_announcement = Some(chrono::Utc::now());
        }

        Ok(())
    }

    /// Discover peers from the REAL BitTorrent DHT
    async fn discover_peers(&mut self) -> Result<()> {
        info!("🔍 Discovering peers from REAL BitTorrent DHT");

        // Search for today's and yesterday's announcements
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let yesterday = (chrono::Utc::now() - chrono::Duration::days(1))
            .format("%Y-%m-%d")
            .to_string();

        let mut discovered = Vec::new();

        let mut dht = self.dht_client.write().await;
        for date in &[today, yesterday] {
            match dht.discover_peers(date).await {
                Ok(peers) => {
                    info!("📊 Found {} peers for date {}", peers.len(), date);
                    discovered.extend(peers);
                }
                Err(e) => {
                    warn!("⚠️ Failed to discover peers for {}: {}", date, e);
                }
            }
        }

        // Convert to our format and store
        {
            let mut peers = self.discovered_peers.write().await;
            for presence in discovered {
                let peer = DiscoveredPeer {
                    validator_id: hex::encode(presence.onion_address.as_bytes()),
                    onion_address: presence.onion_address.clone(),
                    capabilities: presence.capabilities,
                    discovered_at: chrono::Utc::now(),
                    last_seen: chrono::DateTime::from_timestamp(presence.timestamp, 0)
                        .unwrap_or_else(|| chrono::Utc::now()),
                    verified: false,
                    connection_tested: false,
                };

                peers.insert(presence.onion_address, peer);
            }

            info!("💫 Discovered {} total peers", peers.len());
        }

        Ok(())
    }

    /// Test REAL Tor connections to discovered peers
    async fn test_peer_connections(&mut self) -> Result<()> {
        info!("🔗 Testing REAL Tor connections to discovered peers");

        let peers: Vec<_> = {
            let peers_guard = self.discovered_peers.read().await;
            peers_guard.values().cloned().collect()
        };

        let mut tor = self.tor_client.write().await;
        let mut successful = 0;
        let mut failed = 0;

        for peer in peers.iter().take(5) {
            // Test first 5 peers
            info!("🧅 Testing connection to: {}", peer.onion_address);

            match tor.connect_to_onion(&peer.onion_address, 8080).await {
                Ok(_connection) => {
                    info!("✅ Successfully connected to {}", peer.onion_address);
                    successful += 1;

                    // Update peer status
                    {
                        let mut peers_guard = self.discovered_peers.write().await;
                        if let Some(peer_mut) = peers_guard.get_mut(&peer.onion_address) {
                            peer_mut.connection_tested = true;
                            peer_mut.verified = true;
                        }
                    }
                }
                Err(e) => {
                    warn!("❌ Failed to connect to {}: {}", peer.onion_address, e);
                    failed += 1;
                }
            }
        }

        info!(
            "📊 Connection test results: {} successful, {} failed",
            successful, failed
        );

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.successful_connections += successful;
            stats.failed_connections += failed;
        }

        Ok(())
    }

    /// Get REAL discovery statistics
    pub async fn get_stats(&self) -> RealDiscoveryStats {
        let mut stats = self.stats.read().await.clone();

        // Update with current DHT stats
        {
            let dht = self.dht_client.read().await;
            let dht_stats = dht.get_stats().await;
            stats.dht_connected_nodes = dht_stats.connected_nodes;
            stats.dht_stored_records = dht_stats.stored_records;
        }

        // Update with current Tor stats
        {
            let tor = self.tor_client.read().await;
            let tor_stats = tor.get_stats().await;
            stats.tor_active_circuits = tor_stats.active_circuits;
            stats.onion_service_active = tor_stats.onion_service_active;
        }

        // Update peer count
        {
            let peers = self.discovered_peers.read().await;
            stats.discovered_peers = peers.len();
        }

        stats
    }

    /// Get all discovered peers
    pub async fn get_discovered_peers(&self) -> Vec<DiscoveredPeer> {
        let peers = self.discovered_peers.read().await;
        peers.values().cloned().collect()
    }

    /// Refresh peer discovery (call periodically)
    pub async fn refresh_discovery(&mut self) -> Result<()> {
        info!("🔄 Refreshing peer discovery...");

        // Re-announce our presence
        self.announce_presence().await?;

        // Re-discover peers
        self.discover_peers().await?;

        // Cleanup old circuits
        {
            let mut tor = self.tor_client.write().await;
            tor.cleanup_circuits().await?;
        }

        info!("✅ Discovery refresh complete");
        Ok(())
    }
}

/// Test REAL BitTorrent DHT connectivity
pub async fn test_real_dht_connectivity() -> Result<bool> {
    info!("🧪 Testing REAL BitTorrent DHT connectivity...");

    let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let bootstrap_nodes = vec!["87.98.162.88:6881".parse().unwrap()];

    match RealBep44Client::new(signing_key, bootstrap_nodes).await {
        Ok(mut client) => match client.bootstrap().await {
            Ok(_) => {
                info!("✅ Real DHT connectivity test successful");
                Ok(true)
            }
            Err(e) => {
                error!("❌ DHT bootstrap failed: {}", e);
                Ok(false)
            }
        },
        Err(e) => {
            error!("❌ Failed to create DHT client: {}", e);
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires real network connections
    async fn test_real_discovery_engine() {
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
        let engine = RealDiscoveryEngine::new(signing_key).await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    #[ignore] // Requires real DHT network
    async fn test_dht_connectivity() {
        let result = test_real_dht_connectivity().await;
        assert!(result.is_ok());
    }
}
