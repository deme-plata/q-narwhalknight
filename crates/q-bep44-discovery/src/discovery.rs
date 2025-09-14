/*!
# Peer Discovery Engine using BEP-44 DHT

High-level peer discovery orchestration combining all BEP-44 components.

This module coordinates:
- **BEP-44 DHT Client**: For mutable data storage/retrieval
- **Presence Management**: Periodic announcements and key rotation  
- **Crypto Management**: Encryption and signatures
- **Decoy Generation**: Cover traffic for privacy
- **Tor Bridge**: Anonymous transport connections

## Discovery Process

1. **Bootstrap**: Connect to BitTorrent DHT network
2. **Announce**: Store encrypted presence records periodically
3. **Search**: Look for friends using time-based key rotation
4. **Decrypt**: Process discovered announcements from friends
5. **Connect**: Establish Tor connections to discovered peers
6. **Maintain**: Keep presence active and rotate keys

## Integration with Q-NarwhalKnight

The discovery engine integrates seamlessly with the existing Q-NarwhalKnight
architecture, replacing DNS-phantom discovery while maintaining the same
bridge interface to NetworkManager and consensus systems.

```
BEP-44 Discovery → NetworkManager → DAG-Knight Consensus
        ↕                ↕              ↕
   DHT Network     P2P Network    Transaction Processing
```
*/

use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use ed25519_dalek::SigningKey;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::{interval, sleep, Instant};
use tokio::task::JoinHandle;

use crate::{
    bep44::Bep44Client,
    crypto::CryptoManager,
    decoy::DecoyGenerator,
    presence::PresenceManager,
    tor_bridge::TorBridge,
    Bep44DiscoveryConfig, DiscoveredPeer, PeerCapability,
};

/// Main discovery engine orchestrating all BEP-44 components
#[derive(Debug)]
pub struct DiscoveryEngine {
    config: Bep44DiscoveryConfig,
    bep44_client: Option<Bep44Client>,
    crypto_manager: CryptoManager,
    presence_manager: Option<PresenceManager>,
    tor_bridge: Option<TorBridge>,
    decoy_generator: Option<DecoyGenerator>,
    discovered_peers: Arc<RwLock<HashMap<[u8; 32], DiscoveredPeer>>>,
    discovery_stats: Arc<RwLock<DiscoveryStats>>,
    background_tasks: Vec<JoinHandle<()>>,
    is_running: Arc<RwLock<bool>>,
}

impl DiscoveryEngine {
    /// Create new discovery engine
    pub async fn new(config: Bep44DiscoveryConfig) -> Result<Self> {
        tracing::info!("🔍 Creating BEP-44 discovery engine");
        
        let crypto_manager = CryptoManager::new(config.validator_keypair.clone())?;
        
        Ok(Self {
            config,
            bep44_client: None,
            crypto_manager,
            presence_manager: None,
            tor_bridge: None,
            decoy_generator: None,
            discovered_peers: Arc::new(RwLock::new(HashMap::new())),
            discovery_stats: Arc::new(RwLock::new(DiscoveryStats::default())),
            background_tasks: Vec::new(),
            is_running: Arc::new(RwLock::new(false)),
        })
    }
    
    /// Initialize all discovery components
    pub async fn initialize(&mut self) -> Result<()> {
        tracing::info!("🚀 Initializing BEP-44 discovery engine");
        
        // Initialize BEP-44 DHT client
        self.bep44_client = Some(Bep44Client::new(
            self.config.bind_address,
            self.config.validator_keypair.clone(),
            self.config.bootstrap_nodes.clone(),
        ).await?);
        
        // Bootstrap DHT connection
        if let Some(client) = &mut self.bep44_client {
            client.bootstrap().await?;
        }
        
        // Initialize presence manager
        self.presence_manager = Some(PresenceManager::new(
            self.config.validator_keypair.clone(),
            self.config.announcement_interval,
            self.config.key_rotation_interval,
        ).await?);
        
        // Initialize Tor bridge
        self.tor_bridge = Some(TorBridge::new(self.config.tor_socks_proxy).await?);
        
        // Initialize decoy generator if enabled
        if self.config.enable_decoy_traffic {
            self.decoy_generator = Some(DecoyGenerator::new().await?);
        }
        
        tracing::info!("✅ BEP-44 discovery engine initialized");
        Ok(())
    }
    
    /// Start the discovery process
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("🌟 Starting BEP-44 peer discovery");
        
        // Mark as running
        {
            let mut running = self.is_running.write().await;
            *running = true;
        }
        
        // Start presence announcements
        if let Some(presence_manager) = &mut self.presence_manager {
            presence_manager.start_announcements().await?;
        }
        
        // Start decoy traffic generation
        if let Some(decoy_generator) = &mut self.decoy_generator {
            decoy_generator.start().await?;
        }
        
        // Start background tasks
        self.start_background_tasks().await?;
        
        tracing::info!("🚀 BEP-44 discovery engine is running");
        Ok(())
    }
    
    /// Stop the discovery process
    pub async fn stop(&mut self) -> Result<()> {
        tracing::info!("🛑 Stopping BEP-44 discovery engine");
        
        // Mark as not running
        {
            let mut running = self.is_running.write().await;
            *running = false;
        }
        
        // Cancel background tasks
        for task in self.background_tasks.drain(..) {
            task.abort();
        }
        
        tracing::info!("✅ BEP-44 discovery engine stopped");
        Ok(())
    }
    
    /// Add friend for encrypted peer discovery
    pub fn add_friend(&mut self, friend_public_key: [u8; 32], shared_secret: [u8; 32]) {
        self.crypto_manager.add_friend(friend_public_key, shared_secret);
        
        if let Some(presence_manager) = &mut self.presence_manager {
            presence_manager.add_friend(friend_public_key, shared_secret.to_vec());
        }
        
        tracing::info!("👥 Added friend to discovery network: {}", 
                      hex::encode(&friend_public_key[..4]));
    }
    
    /// Get all discovered peers
    pub async fn get_discovered_peers(&self) -> Vec<DiscoveredPeer> {
        let peers = self.discovered_peers.read().await;
        peers.values().cloned().collect()
    }
    
    /// Get discovery statistics
    pub async fn get_discovery_stats(&self) -> DiscoveryStats {
        let stats = self.discovery_stats.read().await;
        stats.clone()
    }
    
    /// Connect to a specific discovered peer
    pub async fn connect_to_peer(&self, validator_id: &[u8; 32]) -> Result<()> {
        let peers = self.discovered_peers.read().await;
        
        if let Some(peer) = peers.get(validator_id) {
            if let Some(tor_bridge) = &self.tor_bridge {
                tor_bridge.connect_to_peer(peer).await?;
                
                // Update stats
                {
                    let mut stats = self.discovery_stats.write().await;
                    stats.successful_connections += 1;
                }
                
                tracing::info!("✅ Connected to peer {} via Tor", 
                              hex::encode(&validator_id[..4]));
                
                Ok(())
            } else {
                anyhow::bail!("Tor bridge not initialized");
            }
        } else {
            anyhow::bail!("Peer not found in discovered peers");
        }
    }
    
    /// Start background tasks
    async fn start_background_tasks(&mut self) -> Result<()> {
        let is_running = Arc::clone(&self.is_running);
        
        // Task 1: Periodic peer discovery
        {
            let discovered_peers = Arc::clone(&self.discovered_peers);
            let discovery_stats = Arc::clone(&self.discovery_stats);
            let presence_manager = self.presence_manager.as_ref().unwrap().clone(); // This would need Arc<Mutex<>> in real code
            let mut bep44_client = self.bep44_client.as_ref().unwrap().clone(); // Same here
            let is_running_clone = Arc::clone(&is_running);
            
            let task = tokio::spawn(async move {
                let mut discovery_interval = interval(Duration::from_secs(300)); // 5 minutes
                
                while *is_running_clone.read().await {
                    discovery_interval.tick().await;
                    
                    // Discover peers using time-based key search
                    match presence_manager.discover_peers(&mut bep44_client, 24).await {
                        Ok(peers) => {
                            let mut discovered = discovered_peers.write().await;
                            let mut stats = discovery_stats.write().await;
                            
                            for peer in peers {
                                let was_new = !discovered.contains_key(&peer.validator_id);
                                discovered.insert(peer.validator_id, peer);
                                
                                if was_new {
                                    stats.total_discovered_peers += 1;
                                    tracing::info!("🔍 Discovered new peer: {}", 
                                                  hex::encode(&peer.validator_id[..4]));
                                }
                            }
                            
                            stats.last_discovery_time = Some(Utc::now());
                        }
                        Err(e) => {
                            tracing::warn!("⚠️ Peer discovery failed: {}", e);
                        }
                    }
                }
            });
            
            self.background_tasks.push(task);
        }
        
        // Task 2: Presence announcements
        {
            let presence_manager = self.presence_manager.as_ref().unwrap().clone();
            let mut bep44_client = self.bep44_client.as_ref().unwrap().clone();
            let is_running_clone = Arc::clone(&is_running);
            let validator_id = self.crypto_manager.get_public_key();
            
            let task = tokio::spawn(async move {
                let mut announcement_interval = interval(Duration::from_secs(300)); // 5 minutes
                
                while *is_running_clone.read().await {
                    announcement_interval.tick().await;
                    
                    // Announce our presence
                    let onion_address = format!("{}.onion", hex::encode(&validator_id[..16]));
                    let capabilities = vec![
                        PeerCapability::Consensus,
                        PeerCapability::Mempool,
                        PeerCapability::StateSync,
                    ];
                    
                    if let Err(e) = presence_manager.announce_presence(
                        &mut bep44_client,
                        validator_id,
                        onion_address,
                        capabilities,
                    ).await {
                        tracing::warn!("⚠️ Presence announcement failed: {}", e);
                    }
                }
            });
            
            self.background_tasks.push(task);
        }
        
        // Task 3: Key rotation
        {
            let presence_manager = self.presence_manager.as_ref().unwrap().clone();
            let is_running_clone = Arc::clone(&is_running);
            
            let task = tokio::spawn(async move {
                let mut rotation_interval = interval(Duration::from_secs(3600)); // 1 hour
                
                while *is_running_clone.read().await {
                    rotation_interval.tick().await;
                    
                    if let Err(e) = presence_manager.rotate_lookup_key().await {
                        tracing::warn!("⚠️ Key rotation failed: {}", e);
                    } else {
                        tracing::info!("🔄 Lookup key rotated for privacy");
                    }
                }
            });
            
            self.background_tasks.push(task);
        }
        
        // Task 4: Decoy traffic generation (if enabled)
        if self.config.enable_decoy_traffic {
            let decoy_generator = self.decoy_generator.as_ref().unwrap().clone();
            let mut bep44_client = self.bep44_client.as_ref().unwrap().clone();
            let is_running_clone = Arc::clone(&is_running);
            
            let task = tokio::spawn(async move {
                if let Err(e) = decoy_generator.run_continuous_decoy_generation(&mut bep44_client).await {
                    tracing::error!("❌ Decoy generation failed: {}", e);
                }
            });
            
            self.background_tasks.push(task);
        }
        
        // Task 5: Connection maintenance
        {
            let tor_bridge = self.tor_bridge.as_ref().unwrap().clone();
            let is_running_clone = Arc::clone(&is_running);
            
            let task = tokio::spawn(async move {
                let mut maintenance_interval = interval(Duration::from_secs(60)); // 1 minute
                
                while *is_running_clone.read().await {
                    maintenance_interval.tick().await;
                    
                    if let Err(e) = tor_bridge.cleanup_inactive_connections().await {
                        tracing::warn!("⚠️ Connection cleanup failed: {}", e);
                    }
                }
            });
            
            self.background_tasks.push(task);
        }
        
        tracing::info!("🔄 Started {} background discovery tasks", self.background_tasks.len());
        Ok(())
    }
    
    /// Force immediate peer discovery
    pub async fn force_discovery(&self) -> Result<Vec<DiscoveredPeer>> {
        tracing::info!("🔍 Forcing immediate peer discovery");
        
        if let (Some(presence_manager), Some(mut bep44_client)) = 
           (&self.presence_manager, self.bep44_client.as_ref().map(|c| c.clone())) {
            
            let discovered = presence_manager.discover_peers(&mut bep44_client, 24).await?;
            
            // Update internal peer list
            {
                let mut peers = self.discovered_peers.write().await;
                for peer in &discovered {
                    peers.insert(peer.validator_id, peer.clone());
                }
            }
            
            tracing::info!("✅ Force discovery completed - Found {} peers", discovered.len());
            Ok(discovered)
        } else {
            anyhow::bail!("Discovery components not initialized");
        }
    }
    
    /// Get Tor bridge statistics
    pub async fn get_tor_stats(&self) -> Option<crate::tor_bridge::TorCircuitStats> {
        if let Some(tor_bridge) = &self.tor_bridge {
            Some(tor_bridge.get_stats().await)
        } else {
            None
        }
    }
}

/// Discovery engine statistics
#[derive(Debug, Clone, Default)]
pub struct DiscoveryStats {
    pub total_discovered_peers: u64,
    pub successful_connections: u64,
    pub failed_connections: u64,
    pub total_announcements: u64,
    pub last_discovery_time: Option<DateTime<Utc>>,
    pub average_discovery_time_ms: u64,
    pub active_background_tasks: u32,
}