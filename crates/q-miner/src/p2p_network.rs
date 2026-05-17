//! P2P Mining Network Module
//!
//! Lightweight libp2p client for miners — subscribes to gossipsub topics for
//! challenge relay, solution broadcast, and block signals. HTTP remains the
//! automatic fallback when P2P is unavailable.
//!
//! v9.1.7: Initial implementation.

use anyhow::{Context, Result};
use futures::StreamExt;
use libp2p::{
    gossipsub, identify, kad, noise, ping, tcp, yamux,
    Multiaddr, PeerId, Swarm, SwarmBuilder,
    swarm::NetworkBehaviour,
};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

use q_types::mining_solution::{NetworkChallenge, P2PMiningMessage, P2PMiningSubmission};

// ═══════════════════════════════════════════════════════════════════
// Bootstrap peers — same 4 hardcoded peers as q-network
// ═══════════════════════════════════════════════════════════════════
const HARDCODED_BOOTSTRAP_PEERS: &[&str] = &[
    // Epsilon - 10Gbit SUPERNODE
    "/ip4/89.149.241.126/tcp/9001/p2p/12D3KooWFpbXxxZJQ4FX9FGXrE5vaeNTCnZmLn6bqToRCMuiMpxM",
    // Delta - 1Gbit
    "/ip4/5.79.79.158/tcp/9001/p2p/12D3KooWLJJRvqo6mBoHLpgxVbGKfW3Jv39ziU4kz1adKFv93JbK",
    // Gamma - 1Gbit
    "/ip4/109.205.176.60/tcp/9001/p2p/12D3KooWFfZKfKbBnB5SehTRBacHndyhJ6aQWxTAQrrwXA7761cH",
    // Beta - 100Mbit (DHT coordinator)
    "/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWSBxwSKw4wftHViMdw5rrV8Z1wEkikDS2vKYZtRrio5hH",
];

// ═══════════════════════════════════════════════════════════════════
// Gossipsub topic prefix — matches q-types NetworkId format
// ═══════════════════════════════════════════════════════════════════
fn gossipsub_topic_prefix(network_id: &str) -> String {
    format!("/qnk/{}", network_id)
}

fn mining_challenges_topic(network_id: &str) -> String {
    format!("{}/mining-challenges", gossipsub_topic_prefix(network_id))
}

fn mining_solutions_topic(network_id: &str) -> String {
    format!("{}/mining-solutions", gossipsub_topic_prefix(network_id))
}

#[allow(dead_code)]
fn blocks_topic(network_id: &str) -> String {
    format!("{}/blocks", gossipsub_topic_prefix(network_id))
}

// ═══════════════════════════════════════════════════════════════════
// MinerBehaviour — minimal libp2p behaviour for miners
// ═══════════════════════════════════════════════════════════════════
#[derive(NetworkBehaviour)]
struct MinerBehaviour {
    gossipsub: gossipsub::Behaviour,
    kademlia: kad::Behaviour<kad::store::MemoryStore>,
    identify: identify::Behaviour,
    ping: ping::Behaviour,
}

// ═══════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════
pub struct MinerP2PConfig {
    pub listen_port: u16,
    pub network_id: String,
    pub bootstrap_peers: Vec<String>,
}

impl Default for MinerP2PConfig {
    fn default() -> Self {
        Self {
            listen_port: 0, // Random port
            network_id: std::env::var("Q_NETWORK_ID")
                .unwrap_or_else(|_| "mainnet-genesis".to_string()),
            bootstrap_peers: Vec::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MinerP2PNetwork — main P2P client
// ═══════════════════════════════════════════════════════════════════
pub struct MinerP2PNetwork {
    swarm: Swarm<MinerBehaviour>,
    local_peer_id: PeerId,
    network_id: String,

    // Channel senders for pushing data to mining threads
    challenge_tx: broadcast::Sender<NetworkChallenge>,
    block_signal_tx: broadcast::Sender<u64>,
    // Channel for receiving solutions to broadcast
    solution_rx: tokio::sync::mpsc::UnboundedReceiver<P2PMiningSubmission>,
    solution_tx_handle: tokio::sync::mpsc::UnboundedSender<P2PMiningSubmission>,

    // Shared state for TUI status
    p2p_connected: Arc<AtomicBool>,
    p2p_peer_count: Arc<AtomicU32>,
    p2p_challenges_received: Arc<AtomicU64>,
    p2p_solutions_broadcast: Arc<AtomicU64>,

    // Dedup: track seen challenge heights and solution hashes
    seen_challenge_height: u64,
    seen_solutions: HashSet<[u8; 32]>,

    // Bootstrap addrs for reconnection
    bootstrap_addrs: Vec<(PeerId, Multiaddr)>,
}

impl MinerP2PNetwork {
    pub async fn new(
        config: MinerP2PConfig,
        p2p_connected: Arc<AtomicBool>,
        p2p_peer_count: Arc<AtomicU32>,
        p2p_challenges_received: Arc<AtomicU64>,
        p2p_solutions_broadcast: Arc<AtomicU64>,
    ) -> Result<Self> {
        // Generate ephemeral keypair (miners don't need persistent identity)
        let local_key = libp2p::identity::Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(local_key.public());
        info!("P2P miner peer ID: {}", local_peer_id);

        // v10.0.2: Miner-optimized gossipsub — minimal mesh, no flood publish.
        // Miners are leaf nodes, not routers. 2 mesh peers is sufficient for
        // challenge relay. HTTP is the reliable path for solution submission
        // (bootstrap nodes already drop P2P mining gossip via Q_SKIP_MINING_GOSSIP=1).
        // flood_publish=false reduces per-publish bandwidth from ALL 80+ peers to mesh only (2).
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_secs(5)) // 5x reduction (was 1s)
            .mesh_n(2)       // Minimal mesh (was 4)
            .mesh_n_low(1)   // Allow 1 peer minimum (was 2)
            .mesh_n_high(4)  // Cap at 4 (was 6)
            .flood_publish(false) // Mesh-only publish (was true → ALL peers)
            .max_transmit_size(1_048_576) // 1 MB
            .validation_mode(gossipsub::ValidationMode::Permissive)
            .build()
            .map_err(|e| anyhow::anyhow!("Gossipsub config error: {}", e))?;

        let gossipsub = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(local_key.clone()),
            gossipsub_config,
        )
        .map_err(|e| anyhow::anyhow!("Gossipsub behaviour error: {}", e))?;

        // Kademlia for DHT-based peer discovery
        let kademlia = kad::Behaviour::new(
            local_peer_id,
            kad::store::MemoryStore::new(local_peer_id),
        );

        // Identify — required for gossipsub peer negotiation
        let identify = identify::Behaviour::new(identify::Config::new(
            format!("/qnk-miner/{}", env!("CARGO_PKG_VERSION")),
            local_key.public(),
        ));

        let ping = ping::Behaviour::default();

        let behaviour = MinerBehaviour {
            gossipsub,
            kademlia,
            identify,
            ping,
        };

        let mut swarm = SwarmBuilder::with_existing_identity(local_key)
            .with_tokio()
            .with_tcp(
                tcp::Config::default().nodelay(true),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_behaviour(|_| Ok(behaviour))
            .map_err(|e| anyhow::anyhow!("Swarm build error: {}", e))?
            .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(300)))
            .build();

        // Listen on configured port (0 = OS-assigned random port)
        let listen_addr: Multiaddr = format!("/ip4/0.0.0.0/tcp/{}", config.listen_port)
            .parse()
            .context("Invalid listen address")?;
        swarm.listen_on(listen_addr)?;

        // Subscribe to mining topics only (no blocks — miners get block signals via SSE).
        // v10.0.2: Removed blocks topic subscription — saves ~10 KB/s inbound bandwidth
        // since full blocks (10-100KB each, 1/sec) are unnecessary for miners.
        let challenges_topic = gossipsub::IdentTopic::new(mining_challenges_topic(&config.network_id));
        let solutions_topic = gossipsub::IdentTopic::new(mining_solutions_topic(&config.network_id));

        swarm.behaviour_mut().gossipsub.subscribe(&challenges_topic)
            .map_err(|e| anyhow::anyhow!("Subscribe challenges: {}", e))?;
        swarm.behaviour_mut().gossipsub.subscribe(&solutions_topic)
            .map_err(|e| anyhow::anyhow!("Subscribe solutions: {}", e))?;

        info!("P2P subscribed to: {}, {} (blocks topic skipped — SSE provides block signals)",
            mining_challenges_topic(&config.network_id),
            mining_solutions_topic(&config.network_id));

        // Parse bootstrap peers
        let mut bootstrap_peers: Vec<String> = config.bootstrap_peers;
        if bootstrap_peers.is_empty() {
            // Use hardcoded + env override
            if let Ok(env_peers) = std::env::var("Q_BOOTSTRAP_PEERS") {
                bootstrap_peers = env_peers.split(',').map(|s| s.trim().to_string()).collect();
            } else {
                bootstrap_peers = HARDCODED_BOOTSTRAP_PEERS.iter().map(|s| s.to_string()).collect();
            }
        }

        let mut bootstrap_addrs = Vec::new();
        for peer_str in &bootstrap_peers {
            if let Ok(addr) = peer_str.parse::<Multiaddr>() {
                if let Some(peer_id) = extract_peer_id(&addr) {
                    // Add to Kademlia routing table
                    swarm.behaviour_mut().kademlia.add_address(&peer_id, addr.clone());
                    bootstrap_addrs.push((peer_id, addr));
                }
            }
        }

        // Dial bootstrap peers
        for (peer_id, addr) in &bootstrap_addrs {
            debug!("P2P dialing bootstrap peer: {} at {}", peer_id, addr);
            if let Err(e) = swarm.dial(addr.clone()) {
                warn!("P2P failed to dial {}: {}", peer_id, e);
            }
        }

        // Bootstrap Kademlia
        if let Err(e) = swarm.behaviour_mut().kademlia.bootstrap() {
            debug!("P2P Kademlia bootstrap: {} (expected if no peers yet)", e);
        }

        // Create channels
        let (challenge_tx, _) = broadcast::channel(64);
        let (block_signal_tx, _) = broadcast::channel(64);
        let (solution_tx, solution_rx) = tokio::sync::mpsc::unbounded_channel();

        Ok(Self {
            swarm,
            local_peer_id,
            network_id: config.network_id,
            challenge_tx,
            block_signal_tx,
            solution_rx,
            solution_tx_handle: solution_tx,
            p2p_connected,
            p2p_peer_count,
            p2p_challenges_received,
            p2p_solutions_broadcast,
            seen_challenge_height: 0,
            seen_solutions: HashSet::new(),
            bootstrap_addrs,
        })
    }

    /// Get a receiver for verified P2P challenges
    pub fn challenge_receiver(&self) -> broadcast::Receiver<NetworkChallenge> {
        self.challenge_tx.subscribe()
    }

    /// Get a receiver for new-block height signals
    pub fn block_signal_receiver(&self) -> broadcast::Receiver<u64> {
        self.block_signal_tx.subscribe()
    }

    /// Get a sender for submitting solutions via P2P
    pub fn solution_sender(&self) -> tokio::sync::mpsc::UnboundedSender<P2PMiningSubmission> {
        self.solution_tx_handle.clone()
    }

    /// Current peer count
    pub fn peer_count(&self) -> usize {
        self.p2p_peer_count.load(Ordering::Relaxed) as usize
    }

    /// Whether connected to any peers
    pub fn is_connected(&self) -> bool {
        self.p2p_connected.load(Ordering::Relaxed)
    }

    /// Run the P2P event loop (spawned as tokio task)
    pub async fn run(mut self) {
        use libp2p::swarm::SwarmEvent;

        let mut reconnect_interval = tokio::time::interval(Duration::from_secs(30));
        let mut warmup_attempts: u32 = 0;
        let mut warmup_done = false;

        // Initial warmup: retry bootstrap with exponential backoff
        let warmup_schedule = [3u64, 6, 12, 30];

        loop {
            tokio::select! {
                // Process swarm events
                event = self.swarm.select_next_some() => {
                    match event {
                        SwarmEvent::Behaviour(MinerBehaviourEvent::Gossipsub(
                            gossipsub::Event::Message { message, .. }
                        )) => {
                            self.handle_gossipsub_message(&message);
                        }
                        SwarmEvent::Behaviour(MinerBehaviourEvent::Gossipsub(
                            gossipsub::Event::Subscribed { peer_id, topic }
                        )) => {
                            debug!("P2P peer {} subscribed to {}", peer_id, topic);
                        }
                        SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                            let count = self.swarm.connected_peers().count() as u32;
                            self.p2p_peer_count.store(count, Ordering::Relaxed);
                            self.p2p_connected.store(count > 0, Ordering::Relaxed);
                            info!("P2P connected to {} (total: {})", &peer_id.to_string()[..12], count);
                            warmup_done = true;
                        }
                        SwarmEvent::ConnectionClosed { peer_id, .. } => {
                            let count = self.swarm.connected_peers().count() as u32;
                            self.p2p_peer_count.store(count, Ordering::Relaxed);
                            self.p2p_connected.store(count > 0, Ordering::Relaxed);
                            debug!("P2P disconnected from {} (total: {})", &peer_id.to_string()[..12], count);
                        }
                        SwarmEvent::NewListenAddr { address, .. } => {
                            info!("P2P listening on {}", address);
                        }
                        _ => {}
                    }
                }

                // Outbound solution broadcast
                Some(solution) = self.solution_rx.recv() => {
                    self.broadcast_solution(solution);
                }

                // Periodic reconnection + Kademlia refresh
                _ = reconnect_interval.tick() => {
                    let count = self.swarm.connected_peers().count();
                    if count == 0 {
                        info!("P2P: no peers connected, re-dialing bootstrap...");
                        for (_, addr) in &self.bootstrap_addrs {
                            let _ = self.swarm.dial(addr.clone());
                        }
                        let _ = self.swarm.behaviour_mut().kademlia.bootstrap();
                    }
                    // Prune old seen solutions (keep last 1000)
                    if self.seen_solutions.len() > 1000 {
                        self.seen_solutions.clear();
                    }
                }
            }

            // Connection warmup for new identity (exponential backoff)
            if !warmup_done && warmup_attempts < warmup_schedule.len() as u32 {
                let count = self.swarm.connected_peers().count();
                if count == 0 {
                    let delay = warmup_schedule[warmup_attempts as usize];
                    debug!("P2P warmup attempt {} — waiting {}s before re-bootstrap", warmup_attempts + 1, delay);
                    tokio::time::sleep(Duration::from_secs(delay)).await;
                    for (_, addr) in &self.bootstrap_addrs {
                        let _ = self.swarm.dial(addr.clone());
                    }
                    let _ = self.swarm.behaviour_mut().kademlia.bootstrap();
                    warmup_attempts += 1;
                } else {
                    warmup_done = true;
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Gossipsub message handler
    // ═══════════════════════════════════════════════════════════════════
    fn handle_gossipsub_message(&mut self, message: &gossipsub::Message) {
        let topic_str = message.topic.as_str();
        let data = &message.data;

        if topic_str == mining_challenges_topic(&self.network_id) {
            self.handle_challenge_message(data);
        } else if topic_str == mining_solutions_topic(&self.network_id) {
            // We don't need to process other miners' solutions — just log
            debug!("P2P: received mining solution ({} bytes)", data.len());
        } else {
            debug!("P2P: ignoring message on topic {} ({} bytes)", topic_str, data.len());
        }
    }

    fn handle_challenge_message(&mut self, data: &[u8]) {
        // Try deserializing as P2PMiningMessage::Challenge
        match rmp_serde::from_slice::<P2PMiningMessage>(data) {
            Ok(P2PMiningMessage::Challenge(challenge)) => {
                self.process_challenge(challenge);
            }
            // Also try direct NetworkChallenge deserialization (some nodes may send raw)
            Err(_) => {
                match rmp_serde::from_slice::<NetworkChallenge>(data) {
                    Ok(challenge) => {
                        self.process_challenge(challenge);
                    }
                    Err(e) => {
                        debug!("P2P: failed to deserialize challenge: {}", e);
                    }
                }
            }
            Ok(_) => {
                debug!("P2P: unexpected message type on challenges topic");
            }
        }
    }

    fn process_challenge(&mut self, challenge: NetworkChallenge) {
        // Dedup: only accept higher height than current
        if challenge.canonical_height <= self.seen_challenge_height {
            return;
        }

        // BLAKE3 verification: recompute challenge hash and compare
        let computed_hash = challenge.compute_challenge_hash();
        if computed_hash != challenge.challenge_hash {
            warn!("P2P: challenge hash mismatch at height {} — rejecting (anti-spoof)",
                  challenge.canonical_height);
            return;
        }

        // Expiry check
        if !challenge.is_valid() {
            debug!("P2P: expired challenge at height {} — skipping", challenge.canonical_height);
            return;
        }

        // Accept the challenge
        self.seen_challenge_height = challenge.canonical_height;
        self.p2p_challenges_received.fetch_add(1, Ordering::Relaxed);

        info!("P2P challenge received: height={}, difficulty={:.8}",
              challenge.canonical_height,
              hex::encode(&challenge.difficulty_target[..4]));

        // Broadcast to mining threads
        let _ = self.challenge_tx.send(challenge);
    }

    // v10.0.2: Block messages no longer received (blocks topic unsubscribed).
    // Kept for potential future re-enablement.
    #[allow(dead_code)]
    fn handle_block_message(&mut self, data: &[u8]) {
        // Extract height from the block message. We don't need the full block —
        // just the height field to signal mining threads to refresh.
        // Try MessagePack first (VersionedBlock), then JSON fallback
        if let Ok(block_value) = rmp_serde::from_slice::<serde_json::Value>(data) {
            if let Some(height) = block_value.get("height")
                .or_else(|| block_value.get("block").and_then(|b| b.get("height")))
                .and_then(|h| h.as_u64())
            {
                debug!("P2P block signal: height={}", height);
                let _ = self.block_signal_tx.send(height);
                return;
            }
        }
        // If we can't parse height, still signal "some new block"
        debug!("P2P block signal (unparsed, {} bytes)", data.len());
        let _ = self.block_signal_tx.send(0);
    }

    fn broadcast_solution(&mut self, solution: P2PMiningSubmission) {
        // Dedup check
        let dedup_id = solution.dedup_id();
        if self.seen_solutions.contains(&dedup_id) {
            return;
        }
        self.seen_solutions.insert(dedup_id);

        let msg = P2PMiningMessage::Solution(solution);
        match rmp_serde::to_vec(&msg) {
            Ok(bytes) => {
                let topic = gossipsub::IdentTopic::new(mining_solutions_topic(&self.network_id));
                match self.swarm.behaviour_mut().gossipsub.publish(topic, bytes) {
                    Ok(_) => {
                        self.p2p_solutions_broadcast.fetch_add(1, Ordering::Relaxed);
                        debug!("P2P solution broadcast OK");
                    }
                    Err(e) => {
                        debug!("P2P solution broadcast failed: {} (HTTP fallback active)", e);
                    }
                }
            }
            Err(e) => {
                warn!("P2P: failed to serialize solution: {}", e);
            }
        }
    }
}

/// Extract PeerId from a multiaddr like /ip4/.../tcp/.../p2p/<peer_id>
fn extract_peer_id(addr: &Multiaddr) -> Option<PeerId> {
    addr.iter().find_map(|proto| {
        if let libp2p::multiaddr::Protocol::P2p(peer_id) = proto {
            Some(peer_id)
        } else {
            None
        }
    })
}
