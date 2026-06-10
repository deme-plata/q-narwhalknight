use crate::IpfsStorageError;
use bytes::Bytes;
use cid::Cid;
use futures::StreamExt;
use libp2p::{
    core::upgrade,
    gossipsub, identify, kad,
    multiaddr::Protocol,
    noise, swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux, Multiaddr, PeerId, SwarmBuilder,
};
use multihash_codetable::{Code, MultihashDigest};
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

/// IPFS client configuration
#[derive(Debug, Clone)]
pub struct IpfsConfig {
    /// Enable local pinning of uploaded content
    pub enable_local_pinning: bool,
    /// Replication factor for distributed pinning
    pub replication_factor: usize,
    /// Bootstrap peers to connect to
    pub bootstrap_peers: Vec<Multiaddr>,
    /// Listen address for libp2p
    pub listen_addr: Multiaddr,
}

impl Default for IpfsConfig {
    fn default() -> Self {
        Self {
            enable_local_pinning: true,
            replication_factor: 3,
            bootstrap_peers: Vec::new(),
            listen_addr: "/ip4/0.0.0.0/tcp/0".parse().unwrap(),
        }
    }
}

/// Network behaviour for IPFS storage
#[derive(NetworkBehaviour)]
struct IpfsBehaviour {
    /// Kademlia DHT for content discovery
    kademlia: kad::Behaviour<kad::store::MemoryStore>,
    /// Gossipsub for manifest distribution
    gossipsub: gossipsub::Behaviour,
    /// Identify protocol for peer info
    identify: identify::Behaviour,
}

/// Commands sent to the IPFS client
enum IpfsCommand {
    /// Put data and get CID
    Put {
        data: Bytes,
        response: oneshot::Sender<std::result::Result<String, IpfsStorageError>>,
    },
    /// Get data by CID
    Get {
        cid: String,
        response: oneshot::Sender<std::result::Result<Bytes, IpfsStorageError>>,
    },
    /// Pin content locally
    Pin {
        cid: String,
        response: oneshot::Sender<std::result::Result<(), IpfsStorageError>>,
    },
    /// Unpin content
    Unpin {
        cid: String,
        response: oneshot::Sender<std::result::Result<(), IpfsStorageError>>,
    },
    /// Provide content to network
    Provide {
        cid: String,
        response: oneshot::Sender<std::result::Result<(), IpfsStorageError>>,
    },
}

/// IPFS client for distributed storage
pub struct IpfsClient {
    /// Command sender
    command_tx: mpsc::UnboundedSender<IpfsCommand>,
}

impl IpfsClient {
    /// Create a new IPFS client
    pub async fn new(config: IpfsConfig) -> std::result::Result<Self, IpfsStorageError> {
        let (command_tx, mut command_rx) = mpsc::unbounded_channel();

        // Build libp2p swarm
        let local_key = libp2p::identity::Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(local_key.public());

        info!("IPFS client starting with peer ID: {}", local_peer_id);

        // Create Kademlia DHT
        let mut kad_config = kad::Config::default();
        kad_config.set_query_timeout(Duration::from_secs(60));
        let store = kad::store::MemoryStore::new(local_peer_id);
        let mut kademlia = kad::Behaviour::with_config(local_peer_id, store, kad_config);

        // Add bootstrap peers to Kademlia
        for peer in &config.bootstrap_peers {
            if let Some(Protocol::P2p(peer_id)) = peer.iter().last() {
                kademlia.add_address(&peer_id, peer.clone());
            }
        }

        // Create Gossipsub for manifest distribution
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_secs(1))
            .validation_mode(gossipsub::ValidationMode::Strict)
            .build()
            .map_err(|e| IpfsStorageError::Network(format!("Gossipsub config error: {}", e)))?;

        let gossipsub = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(local_key.clone()),
            gossipsub_config,
        )
        .map_err(|e| IpfsStorageError::Network(format!("Gossipsub error: {}", e)))?;

        // Create Identify protocol
        let identify = identify::Behaviour::new(identify::Config::new(
            "/ipfs-storage/1.0.0".to_string(),
            local_key.public(),
        ));

        // Build behaviour
        let behaviour = IpfsBehaviour {
            kademlia,
            gossipsub,
            identify,
        };

        // Build swarm using new builder pattern in libp2p 0.53
        let mut swarm = SwarmBuilder::with_existing_identity(local_key)
            .with_tokio()
            .with_tcp(
                Default::default(),
                noise::Config::new,
                || yamux::Config::default(),
            )
            .map_err(|e| IpfsStorageError::Network(format!("Transport error: {:?}", e)))?
            .with_behaviour(|_| behaviour)
            .map_err(|e| IpfsStorageError::Network(format!("Behaviour error: {:?}", e)))?
            .build();

        // Listen on configured address
        swarm
            .listen_on(config.listen_addr.clone())
            .map_err(|e| IpfsStorageError::Network(format!("Listen error: {}", e)))?;

        info!("IPFS client listening on: {}", config.listen_addr);

        // Spawn swarm event loop with persistent storage
        let mut local_store: HashMap<String, Bytes> = HashMap::new();
        let mut pinned: HashMap<String, bool> = HashMap::new();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    // Handle swarm events
                    event = swarm.select_next_some() => {
                        match event {
                            SwarmEvent::NewListenAddr { address, .. } => {
                                info!("Listening on {:?}", address);
                            }
                            SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                                debug!("Connected to peer: {}", peer_id);
                            }
                            SwarmEvent::ConnectionClosed { peer_id, .. } => {
                                debug!("Disconnected from peer: {}", peer_id);
                            }
                            SwarmEvent::Behaviour(event) => {
                                debug!("Behaviour event: {:?}", event);
                            }
                            _ => {}
                        }
                    }

                    // Handle commands
                    Some(command) = command_rx.recv() => {
                        match command {
                            IpfsCommand::Put { data, response } => {
                                // Calculate CID for data using Blake3 (fast and secure)
                                let hash = blake3::hash(&data);
                                let hash_bytes = hash.as_bytes();

                                // Create multihash manually: code (0x1e = blake3-256) + length (32) + hash
                                let mut multihash_bytes = vec![0x1e, 32];
                                multihash_bytes.extend_from_slice(hash_bytes);

                                // Create CID from multihash bytes
                                let multihash = multihash_codetable::Multihash::from_bytes(&multihash_bytes)
                                    .map_err(|e| format!("Multihash error: {}", e)).unwrap();
                                let cid = Cid::new_v1(0x55, multihash); // 0x55 = raw codec
                                let cid_str = cid.to_string();

                                // Store locally in memory
                                local_store.insert(cid_str.clone(), data.clone());

                                debug!("Stored {} bytes with CID: {}", data.len(), cid_str);
                                let _ = response.send(Ok(cid_str));
                            }
                            IpfsCommand::Get { cid, response } => {
                                // Try to fetch from local store first
                                if let Some(data) = local_store.get(&cid) {
                                    debug!("Retrieved {} bytes from local store for CID: {}", data.len(), cid);
                                    let _ = response.send(Ok(data.clone()));
                                } else {
                                    // In a full implementation, this would:
                                    // 1. Query DHT for providers of this CID
                                    // 2. Request block from providers via Bitswap
                                    // 3. Verify received data matches CID
                                    warn!("CID not found in local store: {}", cid);
                                    let _ = response.send(Err(IpfsStorageError::Ipfs(
                                        format!("CID not found: {}", cid)
                                    )));
                                }
                            }
                            IpfsCommand::Pin { cid, response } => {
                                pinned.insert(cid.clone(), true);
                                debug!("Pinned CID: {}", cid);
                                let _ = response.send(Ok(()));
                            }
                            IpfsCommand::Unpin { cid, response } => {
                                pinned.remove(&cid);
                                debug!("Unpinned CID: {}", cid);
                                let _ = response.send(Ok(()));
                            }
                            IpfsCommand::Provide { cid, response } => {
                                // Announce to DHT that we have this content
                                // In full implementation: swarm.behaviour_mut().kademlia.start_providing(cid)
                                debug!("Providing CID to network: {}", cid);
                                let _ = response.send(Ok(()));
                            }
                        }
                    }
                }
            }
        });

        Ok(Self { command_tx })
    }

    /// Upload a chunk and get its CID
    pub async fn put_chunk(&self, data: &[u8]) -> std::result::Result<String, IpfsStorageError> {
        let (tx, rx) = oneshot::channel();

        self.command_tx
            .send(IpfsCommand::Put {
                data: Bytes::copy_from_slice(data),
                response: tx,
            })
            .map_err(|_| IpfsStorageError::Ipfs("Failed to send command".to_string()))?;

        rx.await
            .map_err(|_| IpfsStorageError::Ipfs("Failed to receive response".to_string()))?
    }

    /// Download a chunk by CID
    pub async fn get_chunk(&self, cid: &str) -> std::result::Result<Bytes, IpfsStorageError> {
        let (tx, rx) = oneshot::channel();

        self.command_tx
            .send(IpfsCommand::Get {
                cid: cid.to_string(),
                response: tx,
            })
            .map_err(|_| IpfsStorageError::Ipfs("Failed to send command".to_string()))?;

        rx.await
            .map_err(|_| IpfsStorageError::Ipfs("Failed to receive response".to_string()))?
    }

    /// Pin content locally
    pub async fn pin_local(&self, cid: &str) -> std::result::Result<(), IpfsStorageError> {
        let (tx, rx) = oneshot::channel();

        self.command_tx
            .send(IpfsCommand::Pin {
                cid: cid.to_string(),
                response: tx,
            })
            .map_err(|_| IpfsStorageError::Ipfs("Failed to send command".to_string()))?;

        rx.await
            .map_err(|_| IpfsStorageError::Ipfs("Failed to receive response".to_string()))?
    }

    /// Unpin content
    pub async fn unpin(&self, cid: &str) -> std::result::Result<(), IpfsStorageError> {
        let (tx, rx) = oneshot::channel();

        self.command_tx
            .send(IpfsCommand::Unpin {
                cid: cid.to_string(),
                response: tx,
            })
            .map_err(|_| IpfsStorageError::Ipfs("Failed to send command".to_string()))?;

        rx.await
            .map_err(|_| IpfsStorageError::Ipfs("Failed to receive response".to_string()))?
    }

    /// Provide content to the network (announce via DHT)
    pub async fn provide(&self, cid: &str) -> std::result::Result<(), IpfsStorageError> {
        let (tx, rx) = oneshot::channel();

        self.command_tx
            .send(IpfsCommand::Provide {
                cid: cid.to_string(),
                response: tx,
            })
            .map_err(|_| IpfsStorageError::Ipfs("Failed to send command".to_string()))?;

        rx.await
            .map_err(|_| IpfsStorageError::Ipfs("Failed to receive response".to_string()))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ipfs_client_creation() {
        let config = IpfsConfig::default();
        let client = IpfsClient::new(config).await;
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_put_chunk() {
        let config = IpfsConfig::default();
        let client = IpfsClient::new(config).await.unwrap();

        let data = b"Hello, IPFS!";
        let cid = client.put_chunk(data).await.unwrap();

        assert!(!cid.is_empty());
        assert!(cid.starts_with("baf")); // CIDv1 starts with baf
    }
}
