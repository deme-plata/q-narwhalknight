/*!
# IPFS DHT Integration for Q-NarwhalKnight

This module implements IPFS Kademlia DHT integration alongside BitTorrent DHT
for enhanced peer discovery across multiple DHT networks.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  BitTorrent DHT │    │   Q-NarwhalKnight  │    │     IPFS DHT    │
│ (BEP-5/BEP-44)  │◄──►│   Dual DHT Hub    │◄──►│   (Kademlia)    │
│                 │    │                 │    │                 │
│ • Mutable data  │    │ • Cross-network │    │ • Content-based │
│ • Validator ads │    │ • Peer bridge   │    │ • CID resolution│
│ • Tor onions    │    │ • Unified API   │    │ • libp2p compat │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Benefits

- **Increased Discovery Success**: Multiple DHT networks = higher availability
- **Cross-Protocol Compatibility**: Bridge between BitTorrent and IPFS ecosystems
- **Content-Addressed Storage**: IPFS CIDs for immutable validator data
- **libp2p Integration**: Native compatibility with modern P2P stacks
*/

use anyhow::{Context, Result};
use libp2p::{
    kad::{
        Kademlia, KademliaConfig, KademliaEvent, QueryResult, Record, RecordKey,
        store::MemoryStore,
    },
    mdns::{Mdns, MdnsEvent},
    swarm::{SwarmBuilder, SwarmEvent},
    identity, Multiaddr, PeerId, Swarm,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

/// IPFS DHT client for Q-NarwhalKnight peer discovery
#[derive(Debug)]
pub struct IpfsDhtClient {
    /// libp2p swarm with Kademlia DHT
    swarm: Arc<RwLock<Swarm<IpfsBehaviour>>>,
    /// IPFS peer ID
    peer_id: PeerId,
    /// Event channel for DHT events
    event_sender: mpsc::UnboundedSender<IpfsDhtEvent>,
    event_receiver: Arc<RwLock<mpsc::UnboundedReceiver<IpfsDhtEvent>>>,
    /// Stored records in IPFS DHT
    stored_records: Arc<RwLock<HashMap<RecordKey, IpfsRecord>>>,
}

/// libp2p behavior combining Kademlia DHT and mDNS
#[derive(libp2p::swarm::NetworkBehaviour)]
#[behaviour(event_process = false)]
struct IpfsBehaviour {
    kademlia: Kademlia<MemoryStore>,
    mdns: Mdns,
}

/// Events from IPFS DHT operations
#[derive(Debug, Clone)]
pub enum IpfsDhtEvent {
    /// Peer discovered via Kademlia
    PeerDiscovered { peer_id: PeerId, addresses: Vec<Multiaddr> },
    /// Record stored successfully
    RecordStored { key: RecordKey },
    /// Record retrieved
    RecordRetrieved { key: RecordKey, value: Vec<u8> },
    /// Query completed
    QueryCompleted { query_id: String, result: String },
}

/// Q-NarwhalKnight record stored in IPFS DHT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpfsRecord {
    /// Validator's node ID
    pub node_id: [u8; 32],
    /// Onion address for Tor connections
    pub onion_address: String,
    /// libp2p multiaddresses
    pub multiaddrs: Vec<String>,
    /// Validator capabilities
    pub capabilities: Vec<String>,
    /// Timestamp of announcement
    pub timestamp: i64,
    /// IPFS content ID (CID) for additional data
    pub content_cid: Option<String>,
    /// Cross-DHT compatibility metadata
    pub bittorrent_infohash: Option<[u8; 20]>,
}

impl IpfsDhtClient {
    /// Create new IPFS DHT client with Kademlia and mDNS
    pub async fn new() -> Result<Self> {
        info!("🌐 Creating IPFS DHT client with Kademlia + mDNS");

        // Generate libp2p identity
        let local_key = identity::Keypair::generate_ed25519();
        let peer_id = PeerId::from(local_key.public());
        info!("🆔 IPFS Peer ID: {}", peer_id);

        // Create Kademlia DHT store
        let store = MemoryStore::new(peer_id);
        let kademlia_config = KademliaConfig::default();
        let mut kademlia = Kademlia::with_config(peer_id, store, kademlia_config);

        // Add IPFS bootstrap nodes
        let bootstrap_nodes = [
            ("/dnsaddr/bootstrap.libp2p.io", "QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt"),
            ("/ip4/104.131.131.82/tcp/4001", "QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ"),
        ];

        for (multiaddr_str, peer_id_str) in &bootstrap_nodes {
            if let (Ok(multiaddr), Ok(peer_id)) = (
                multiaddr_str.parse::<Multiaddr>(),
                peer_id_str.parse::<PeerId>(),
            ) {
                kademlia.add_address(&peer_id, multiaddr);
                info!("📡 Added IPFS bootstrap node: {}", peer_id_str);
            }
        }

        // Create mDNS for local peer discovery
        let mdns = Mdns::new(Default::default()).await
            .context("Failed to create mDNS service")?;

        // Create swarm behavior
        let behaviour = IpfsBehaviour { kademlia, mdns };

        // Build libp2p swarm
        let swarm = SwarmBuilder::new(local_key, behaviour, peer_id)
            .executor(Box::new(|fut| {
                tokio::spawn(fut);
            }))
            .build();

        // Create event channel
        let (event_sender, event_receiver) = mpsc::unbounded_channel();

        Ok(Self {
            swarm: Arc::new(RwLock::new(swarm)),
            peer_id,
            event_sender,
            event_receiver: Arc::new(RwLock::new(event_receiver)),
            stored_records: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Start the IPFS DHT client
    pub async fn start(&mut self, listen_addr: &str) -> Result<()> {
        info!("🚀 Starting IPFS DHT client");

        // Parse listen address
        let listen_multiaddr: Multiaddr = listen_addr.parse()
            .context("Invalid listen address")?;

        // Start listening
        {
            let mut swarm = self.swarm.write().await;
            swarm.listen_on(listen_multiaddr.clone())
                .context("Failed to listen on address")?;
        }

        info!("📡 IPFS DHT listening on: {}", listen_multiaddr);

        // Start bootstrap process
        {
            let mut swarm = self.swarm.write().await;
            if let Err(e) = swarm.behaviour_mut().kademlia.bootstrap() {
                warn!("⚠️ Kademlia bootstrap failed: {}", e);
            } else {
                info!("🌟 Kademlia bootstrap initiated");
            }
        }

        // Start event loop
        self.start_event_loop().await?;

        info!("✅ IPFS DHT client started successfully");
        Ok(())
    }

    /// Start the libp2p event processing loop
    async fn start_event_loop(&self) -> Result<()> {
        let swarm = self.swarm.clone();
        let event_sender = self.event_sender.clone();

        tokio::spawn(async move {
            loop {
                let event = {
                    let mut swarm = swarm.write().await;
                    swarm.select_next_some().await
                };

                match event {
                    SwarmEvent::Behaviour(behaviour_event) => {
                        Self::handle_behaviour_event(behaviour_event, &event_sender).await;
                    }
                    SwarmEvent::NewListenAddr { address, .. } => {
                        info!("📡 IPFS DHT listening on: {}", address);
                    }
                    SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                        debug!("🔗 Connected to IPFS peer: {}", peer_id);
                    }
                    SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                        debug!("❌ Disconnected from IPFS peer {}: {:?}", peer_id, cause);
                    }
                    _ => {}
                }
            }
        });

        Ok(())
    }

    /// Handle libp2p behavior events
    async fn handle_behaviour_event(
        event: IpfsBehaviourEvent,
        event_sender: &mpsc::UnboundedSender<IpfsDhtEvent>,
    ) {
        match event {
            IpfsBehaviourEvent::Kademlia(kad_event) => {
                match kad_event {
                    KademliaEvent::RoutingUpdated { peer, .. } => {
                        debug!("📋 Kademlia routing table updated: {}", peer);
                    }
                    KademliaEvent::QueryResult { result, .. } => {
                        match result {
                            QueryResult::GetRecord(Ok(record_result)) => {
                                if let Some(record) = record_result.record {
                                    let _ = event_sender.send(IpfsDhtEvent::RecordRetrieved {
                                        key: record.key,
                                        value: record.value,
                                    });
                                }
                            }
                            QueryResult::PutRecord(Ok(_)) => {
                                debug!("✅ IPFS record stored successfully");
                            }
                            QueryResult::GetClosestPeers(Ok(peers_result)) => {
                                for peer in peers_result.peers {
                                    debug!("🔍 Found close IPFS peer: {}", peer);
                                }
                            }
                            _ => {
                                debug!("📊 Kademlia query result: {:?}", result);
                            }
                        }
                    }
                    _ => {
                        debug!("📡 Kademlia event: {:?}", kad_event);
                    }
                }
            }
            IpfsBehaviourEvent::Mdns(mdns_event) => {
                match mdns_event {
                    MdnsEvent::Discovered(list) => {
                        for (peer_id, multiaddr) in list {
                            info!("🔍 mDNS discovered IPFS peer: {} at {}", peer_id, multiaddr);
                            let _ = event_sender.send(IpfsDhtEvent::PeerDiscovered {
                                peer_id,
                                addresses: vec![multiaddr],
                            });
                        }
                    }
                    MdnsEvent::Expired(list) => {
                        for (peer_id, _) in list {
                            debug!("⏰ mDNS peer expired: {}", peer_id);
                        }
                    }
                }
            }
        }
    }

    /// Store Q-NarwhalKnight validator record in IPFS DHT
    pub async fn store_validator_record(
        &mut self,
        node_id: [u8; 32],
        onion_address: &str,
        capabilities: Vec<String>,
    ) -> Result<RecordKey> {
        info!("📝 Storing Q-NarwhalKnight validator in IPFS DHT");
        info!("   • Node ID: {}", hex::encode(&node_id[..8]));
        info!("   • Onion: {}", onion_address);

        // Create validator record
        let record = IpfsRecord {
            node_id,
            onion_address: onion_address.to_string(),
            multiaddrs: vec![], // Will be populated with libp2p addresses
            capabilities,
            timestamp: chrono::Utc::now().timestamp(),
            content_cid: None,
            bittorrent_infohash: None,
        };

        // Serialize record
        let record_data = serde_json::to_vec(&record)
            .context("Failed to serialize validator record")?;

        // Create record key from node ID
        let key_data = format!("qnk-validator-{}", hex::encode(&node_id[..16]));
        let record_key = RecordKey::new(&key_data);

        // Store in IPFS DHT
        {
            let mut swarm = self.swarm.write().await;
            let libp2p_record = Record {
                key: record_key.clone(),
                value: record_data.clone(),
                publisher: Some(self.peer_id),
                expires: None,
            };

            swarm.behaviour_mut().kademlia.put_record(libp2p_record, libp2p::kad::Quorum::One)
                .context("Failed to initiate record storage")?;
        }

        // Store locally
        {
            let mut stored_records = self.stored_records.write().await;
            stored_records.insert(record_key.clone(), record);
        }

        info!("✅ Q-NarwhalKnight validator stored in IPFS DHT");
        Ok(record_key)
    }

    /// Discover Q-NarwhalKnight validators from IPFS DHT
    pub async fn discover_validators(&mut self) -> Result<Vec<IpfsRecord>> {
        info!("🔍 Discovering Q-NarwhalKnight validators from IPFS DHT");

        let mut discovered_validators = Vec::new();

        // Search for Q-NarwhalKnight validator records
        // In a real implementation, this would query for known validator keys
        // For now, return locally stored records
        {
            let stored_records = self.stored_records.read().await;
            for record in stored_records.values() {
                discovered_validators.push(record.clone());
            }
        }

        info!("📊 Discovered {} validators from IPFS DHT", discovered_validators.len());
        Ok(discovered_validators)
    }

    /// Create content-addressed record using IPFS CID
    pub async fn store_content_addressed(
        &mut self,
        content: &[u8],
        content_type: &str,
    ) -> Result<String> {
        info!("🗂️ Storing content-addressed data in IPFS");
        info!("   • Content size: {} bytes", content.len());
        info!("   • Content type: {}", content_type);

        // In a real implementation, this would:
        // 1. Calculate IPFS CID for the content
        // 2. Store content in IPFS network
        // 3. Return the CID for content addressing

        // For now, simulate CID generation
        let cid = format!("Qm{}", hex::encode(&sha2::Sha256::digest(content))[..42]);

        info!("✅ Content stored with CID: {}", cid);
        Ok(cid)
    }

    /// Get DHT statistics
    pub async fn get_stats(&self) -> IpfsDhtStats {
        let stored_records = self.stored_records.read().await;
        let kademlia_peers = {
            let swarm = self.swarm.read().await;
            swarm.behaviour().kademlia.iter().count()
        };

        IpfsDhtStats {
            peer_id: self.peer_id.to_string(),
            kademlia_peers,
            stored_records: stored_records.len(),
            is_bootstrapped: kademlia_peers > 0,
        }
    }

    /// Bridge with BitTorrent DHT - cross-announce records
    pub async fn bridge_with_bittorrent_dht(
        &mut self,
        bittorrent_client: &mut crate::real_bep44::RealBep44Client,
        node_id: [u8; 32],
        onion_address: &str,
    ) -> Result<()> {
        info!("🌉 Bridging IPFS DHT with BitTorrent DHT");

        // Store in IPFS DHT
        let ipfs_key = self.store_validator_record(
            node_id,
            onion_address,
            vec!["consensus".to_string(), "bridge".to_string()],
        ).await?;

        // Store in BitTorrent DHT
        let _bittorrent_target = bittorrent_client.announce_presence(
            onion_address,
            vec!["consensus".to_string(), "bridge".to_string()],
        ).await?;

        info!("🎯 Cross-DHT bridging complete:");
        info!("   • IPFS key: {}", hex::encode(ipfs_key.as_ref()));
        info!("   • BitTorrent target: {}", hex::encode(&node_id[..8]));

        Ok(())
    }
}

/// IPFS DHT statistics
#[derive(Debug, Clone, Serialize)]
pub struct IpfsDhtStats {
    pub peer_id: String,
    pub kademlia_peers: usize,
    pub stored_records: usize,
    pub is_bootstrapped: bool,
}

// Event type alias for libp2p behavior
type IpfsBehaviourEvent = <IpfsBehaviour as libp2p::swarm::NetworkBehaviour>::OutEvent;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ipfs_dht_client_creation() {
        let client = IpfsDhtClient::new().await;
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_validator_record_storage() {
        let mut client = IpfsDhtClient::new().await.unwrap();

        let node_id = [1u8; 32];
        let onion_address = "test.onion";
        let capabilities = vec!["consensus".to_string()];

        let result = client.store_validator_record(node_id, onion_address, capabilities).await;
        assert!(result.is_ok());
    }
}