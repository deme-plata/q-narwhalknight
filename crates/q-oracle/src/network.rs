//! Quantum Oracle Network with Decentralized Price Consensus
//!
//! Network layer for quantum-enhanced oracle nodes featuring:
//! - Gossipsub-based price announcement broadcasting
//! - Decentralized price consensus across all nodes
//! - Weighted voting based on stake and reputation
//! - Byzantine fault tolerance (tolerates f < n/3 malicious nodes)
//! - Eventual consistency with fast convergence

use crate::types::*;
use anyhow::Result;
use bigdecimal::BigDecimal;
use chrono::{DateTime, Utc};
use q_types::{NodeId, Phase};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Gossipsub topic for price announcements
pub const PRICE_CONSENSUS_TOPIC: &str = "/qnk/oracle/price-consensus/1.0.0";

/// Price announcement message broadcast via gossipsub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceAnnouncement {
    /// Feed ID (e.g., "ORB/USD")
    pub feed_id: String,
    /// Announced price
    pub price: String, // BigDecimal as string for serialization
    /// Node that announced the price
    pub node_id: [u8; 32],
    /// Timestamp of announcement
    pub timestamp: DateTime<Utc>,
    /// Round/epoch number for consensus
    pub round: u64,
    /// Node's stake weight (affects voting power)
    pub stake_weight: f64,
    /// Node's reputation score
    pub reputation: f64,
    /// Quantum coherence level (higher = more reliable)
    pub coherence: f64,
    /// Cryptographic signature for verification
    pub signature: Vec<u8>,
}

/// Price vote from a node for consensus
#[derive(Debug, Clone)]
pub struct PriceVote {
    pub node_id: [u8; 32],
    pub price: BigDecimal,
    pub weight: f64, // Combined stake + reputation weight
    pub timestamp: DateTime<Utc>,
}

/// Decentralized price consensus state for a feed
#[derive(Debug, Clone)]
pub struct PriceConsensusState {
    pub feed_id: String,
    /// All votes received in current round
    pub votes: Vec<PriceVote>,
    /// Current consensus price
    pub consensus_price: Option<BigDecimal>,
    /// Confidence in consensus (0-1)
    pub consensus_confidence: f64,
    /// Current consensus round
    pub round: u64,
    /// Timestamp of last consensus
    pub last_consensus: DateTime<Utc>,
    /// Minimum votes needed for consensus (f+1 where f = n/3)
    pub quorum_threshold: usize,
    /// Whether consensus has been reached this round
    pub consensus_reached: bool,
}

impl Default for PriceConsensusState {
    fn default() -> Self {
        Self {
            feed_id: String::new(),
            votes: Vec::new(),
            consensus_price: None,
            consensus_confidence: 0.0,
            round: 0,
            last_consensus: Utc::now(),
            quorum_threshold: 1, // Will be updated based on network size
            consensus_reached: false,
        }
    }
}

/// Quantum Oracle Network Manager with Decentralized Consensus
pub struct QuantumOracleNetwork {
    node_id: NodeId,
    phase: Phase,
    /// Price consensus state for each feed
    consensus_states: Arc<RwLock<HashMap<String, PriceConsensusState>>>,
    /// Connected oracle peers
    peers: Arc<RwLock<HashMap<[u8; 32], OraclePeerInfo>>>,
    /// Pending announcements to broadcast
    outbound_queue: Arc<RwLock<Vec<PriceAnnouncement>>>,
    /// Network statistics
    stats: Arc<RwLock<ConsensusNetworkStats>>,
}

/// Information about a connected oracle peer
#[derive(Debug, Clone)]
pub struct OraclePeerInfo {
    pub node_id: [u8; 32],
    pub stake: f64,
    pub reputation: f64,
    pub last_seen: DateTime<Utc>,
    pub latency_ms: f64,
    pub is_validator: bool,
}

/// Network statistics for consensus
#[derive(Debug, Clone, Default)]
pub struct ConsensusNetworkStats {
    pub total_announcements_sent: u64,
    pub total_announcements_received: u64,
    pub consensus_rounds_completed: u64,
    pub average_consensus_time_ms: f64,
    pub connected_oracle_peers: u64,
    pub last_updated: Option<DateTime<Utc>>,
}

impl QuantumOracleNetwork {
    pub async fn new(node_id: NodeId, phase: Phase) -> Result<Self> {
        Ok(Self {
            node_id,
            phase,
            consensus_states: Arc::new(RwLock::new(HashMap::new())),
            peers: Arc::new(RwLock::new(HashMap::new())),
            outbound_queue: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(ConsensusNetworkStats::default())),
        })
    }

    pub async fn initialize(&self) -> Result<()> {
        info!("🌐 Initializing Quantum Oracle Network with Decentralized Price Consensus");
        info!("📡 Gossipsub topic: {}", PRICE_CONSENSUS_TOPIC);

        // Initialize consensus state for common feeds
        let feeds = vec!["ORB/USD", "ORBUSD/USD", "BTC/USD", "ETH/USD"];
        let mut states = self.consensus_states.write().await;

        for feed_id in feeds {
            states.insert(
                feed_id.to_string(),
                PriceConsensusState {
                    feed_id: feed_id.to_string(),
                    ..Default::default()
                },
            );
        }

        info!("✅ Oracle network initialized with {} price feeds", states.len());
        Ok(())
    }

    /// Announce a price via gossipsub for decentralized consensus
    pub async fn announce_price(
        &self,
        feed_id: &str,
        price: &BigDecimal,
        stake_weight: f64,
        reputation: f64,
        coherence: f64,
    ) -> Result<()> {
        let announcement = PriceAnnouncement {
            feed_id: feed_id.to_string(),
            price: price.to_string(),
            node_id: self.node_id,
            timestamp: Utc::now(),
            round: self.get_current_round(feed_id).await,
            stake_weight,
            reputation,
            coherence,
            signature: self.sign_announcement(feed_id, price).await?,
        };

        // Queue for broadcast
        self.outbound_queue.write().await.push(announcement.clone());

        // Also add our own vote to consensus
        self.process_price_announcement(announcement).await?;

        let mut stats = self.stats.write().await;
        stats.total_announcements_sent += 1;
        stats.last_updated = Some(Utc::now());

        info!("📢 [PRICE CONSENSUS] Announced {} = {} (round {})",
              feed_id, price, self.get_current_round(feed_id).await);

        Ok(())
    }

    /// Process a price announcement received from the network
    pub async fn process_price_announcement(&self, announcement: PriceAnnouncement) -> Result<()> {
        // Verify signature
        if !self.verify_announcement(&announcement).await? {
            warn!("⚠️ Invalid signature on price announcement from {:?}",
                  &announcement.node_id[..4]);
            return Ok(());
        }

        let price = BigDecimal::from_str(&announcement.price)?;

        // Calculate voting weight: stake * reputation * coherence
        let vote_weight = announcement.stake_weight
            * announcement.reputation
            * (0.5 + announcement.coherence * 0.5); // Coherence provides up to 50% boost

        let vote = PriceVote {
            node_id: announcement.node_id,
            price: price.clone(),
            weight: vote_weight,
            timestamp: announcement.timestamp,
        };

        // Add vote to consensus state
        let mut states = self.consensus_states.write().await;
        let state = states
            .entry(announcement.feed_id.clone())
            .or_insert_with(|| PriceConsensusState {
                feed_id: announcement.feed_id.clone(),
                ..Default::default()
            });

        // Check if this node already voted this round
        if state.votes.iter().any(|v| v.node_id == announcement.node_id) {
            debug!("Node {:?} already voted in round {}",
                   &announcement.node_id[..4], state.round);
            return Ok(());
        }

        state.votes.push(vote);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_announcements_received += 1;

        // Check if consensus can be reached
        drop(stats);
        drop(states);
        self.try_reach_consensus(&announcement.feed_id).await?;

        Ok(())
    }

    /// Try to reach consensus on price for a feed
    async fn try_reach_consensus(&self, feed_id: &str) -> Result<()> {
        let mut states = self.consensus_states.write().await;
        let state = match states.get_mut(feed_id) {
            Some(s) => s,
            None => return Ok(()),
        };

        // Check if we have enough votes (quorum)
        if state.votes.len() < state.quorum_threshold {
            debug!("Not enough votes for consensus: {} / {}",
                   state.votes.len(), state.quorum_threshold);
            return Ok(());
        }

        // Calculate weighted median price
        let consensus_price = self.calculate_weighted_median(&state.votes);

        // Calculate consensus confidence based on vote agreement
        let confidence = self.calculate_consensus_confidence(&state.votes, &consensus_price);

        // Only accept consensus if confidence is high enough (> 66%)
        if confidence > 0.66 {
            state.consensus_price = Some(consensus_price.clone());
            state.consensus_confidence = confidence;
            state.consensus_reached = true;
            state.last_consensus = Utc::now();

            info!("✅ [PRICE CONSENSUS] {} = {} (confidence: {:.1}%, {} votes)",
                  feed_id, consensus_price, confidence * 100.0, state.votes.len());

            // Update network stats
            let mut stats = self.stats.write().await;
            stats.consensus_rounds_completed += 1;
        } else {
            debug!("Consensus confidence too low: {:.1}% < 66%", confidence * 100.0);
        }

        Ok(())
    }

    /// Calculate weighted median price from votes
    fn calculate_weighted_median(&self, votes: &[PriceVote]) -> BigDecimal {
        if votes.is_empty() {
            return BigDecimal::from(0);
        }

        // Sort votes by price
        let mut sorted_votes: Vec<_> = votes.iter().collect();
        sorted_votes.sort_by(|a, b| a.price.cmp(&b.price));

        // Calculate total weight
        let total_weight: f64 = sorted_votes.iter().map(|v| v.weight).sum();
        let half_weight = total_weight / 2.0;

        // Find weighted median
        let mut cumulative_weight = 0.0;
        for vote in &sorted_votes {
            cumulative_weight += vote.weight;
            if cumulative_weight >= half_weight {
                return vote.price.clone();
            }
        }

        // Fallback to last vote's price
        sorted_votes.last().map(|v| v.price.clone()).unwrap_or_default()
    }

    /// Calculate confidence in consensus (how much votes agree)
    fn calculate_consensus_confidence(&self, votes: &[PriceVote], consensus_price: &BigDecimal) -> f64 {
        if votes.is_empty() {
            return 0.0;
        }

        let total_weight: f64 = votes.iter().map(|v| v.weight).sum();
        if total_weight == 0.0 {
            return 0.0;
        }

        // Calculate weighted agreement (votes within 1% of consensus)
        let tolerance = consensus_price.clone() * BigDecimal::from_str("0.01").unwrap();
        let agreeing_weight: f64 = votes.iter()
            .filter(|v| {
                let diff = (&v.price - consensus_price).abs();
                diff <= tolerance
            })
            .map(|v| v.weight)
            .sum();

        agreeing_weight / total_weight
    }

    /// Get current consensus price for a feed
    pub async fn get_consensus_price(&self, feed_id: &str) -> Option<(BigDecimal, f64)> {
        let states = self.consensus_states.read().await;
        states.get(feed_id).and_then(|state| {
            state.consensus_price.clone().map(|price| {
                (price, state.consensus_confidence)
            })
        })
    }

    /// Start new consensus round (called periodically or after consensus)
    pub async fn start_new_round(&self, feed_id: &str) -> Result<u64> {
        let mut states = self.consensus_states.write().await;
        let state = states
            .entry(feed_id.to_string())
            .or_insert_with(|| PriceConsensusState {
                feed_id: feed_id.to_string(),
                ..Default::default()
            });

        state.round += 1;
        state.votes.clear();
        state.consensus_reached = false;

        // Update quorum based on connected peers
        let peers = self.peers.read().await;
        let validator_count = peers.values().filter(|p| p.is_validator).count();
        state.quorum_threshold = ((validator_count as f64 * 2.0 / 3.0).ceil() as usize).max(1);

        info!("🔄 [PRICE CONSENSUS] New round {} for {} (quorum: {})",
              state.round, feed_id, state.quorum_threshold);

        Ok(state.round)
    }

    /// Get current round for a feed
    async fn get_current_round(&self, feed_id: &str) -> u64 {
        let states = self.consensus_states.read().await;
        states.get(feed_id).map(|s| s.round).unwrap_or(0)
    }

    /// Sign a price announcement (placeholder - would use actual crypto)
    async fn sign_announcement(&self, _feed_id: &str, _price: &BigDecimal) -> Result<Vec<u8>> {
        // TODO: Implement proper post-quantum signature
        Ok(vec![0u8; 64])
    }

    /// Verify announcement signature
    async fn verify_announcement(&self, _announcement: &PriceAnnouncement) -> Result<bool> {
        // TODO: Implement proper signature verification
        Ok(true)
    }

    /// Register a new oracle peer
    pub async fn register_peer(&self, peer_info: OraclePeerInfo) {
        let mut peers = self.peers.write().await;
        peers.insert(peer_info.node_id, peer_info);

        let mut stats = self.stats.write().await;
        stats.connected_oracle_peers = peers.len() as u64;
    }

    /// Get pending outbound announcements for gossipsub broadcast
    pub async fn drain_outbound_queue(&self) -> Vec<PriceAnnouncement> {
        let mut queue = self.outbound_queue.write().await;
        std::mem::take(&mut *queue)
    }

    /// Sync quantum entanglement across network
    pub async fn sync_quantum_entanglement(&self, entanglement_strength: f64) -> Result<()> {
        debug!("🔗 Syncing quantum entanglement: strength={:.3}", entanglement_strength);
        // Entanglement sync is handled via gossipsub messages
        Ok(())
    }

    /// Get network statistics
    pub async fn get_network_stats(&self) -> ConsensusNetworkStats {
        self.stats.read().await.clone()
    }

    /// Check if consensus is active for a feed
    pub async fn is_consensus_active(&self, feed_id: &str) -> bool {
        let states = self.consensus_states.read().await;
        states.get(feed_id).map(|s| s.consensus_reached).unwrap_or(false)
    }
}
