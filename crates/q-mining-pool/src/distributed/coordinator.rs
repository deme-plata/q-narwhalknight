//! Distributed Pool Coordinator
//!
//! Central coordinator that integrates all distributed components:
//! - Share propagation via gossipsub
//! - CRDT-based PPLNS synchronization
//! - Block consensus with multi-node attestation
//! - Threshold signature payouts
//! - Pool node discovery via DHT

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

use super::block_consensus::{
    AggregatedBlockConfirmation, BlockConsensusManager, BlockFoundAnnouncement, ConsensusStatus,
    NodeAttestation,
};
use super::discovery::{PoolNodeDiscovery, PoolNodeInfo};
use super::payout_consensus::{
    PayoutBatch, PayoutConsensusManager, PayoutStatus, PayoutVote, SignedPayoutTransaction,
};
use super::pplns_crdt::{DistributedPPLNS, PPLNSStateHash};
use super::share::DistributedShare;
use super::topics::PoolTopics;
use super::{
    BlockTemplateMessage, DistributedError, DistributedResult, PPLNSSyncMessage, PeerIdBytes,
    PoolHeartbeat, PoolMessage,
};

/// Configuration for the distributed pool coordinator
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Our peer ID
    pub peer_id: PeerIdBytes,

    /// Network ID
    pub network_id: String,

    /// Stratum server port
    pub stratum_port: u16,

    /// Geographic region
    pub region: String,

    /// Heartbeat interval
    pub heartbeat_interval: Duration,

    /// PPLNS sync interval
    pub pplns_sync_interval: Duration,

    /// Discovery interval
    pub discovery_interval: Duration,

    /// Our signer index for threshold signatures
    pub signer_index: u32,

    /// Our secret share for threshold signatures
    pub secret_share: [u8; 32],

    /// Minimum attestations for block consensus
    pub min_attestations: usize,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            peer_id: [0u8; 32],
            network_id: "testnet-phase16".to_string(),
            stratum_port: 3333,
            region: "unknown".to_string(),
            heartbeat_interval: Duration::from_secs(30),
            pplns_sync_interval: Duration::from_secs(60),
            discovery_interval: Duration::from_secs(120),
            signer_index: 0,
            secret_share: [0u8; 32],
            min_attestations: 3,
        }
    }
}

/// Events emitted by the coordinator
#[derive(Debug, Clone)]
pub enum CoordinatorEvent {
    /// New share received and validated
    ShareReceived {
        share_id: [u8; 32],
        worker: String,
        difficulty: f64,
    },

    /// Block found and announced
    BlockFound {
        block_hash: [u8; 32],
        height: u64,
        finder: String,
    },

    /// Block consensus reached
    BlockConfirmed {
        block_hash: [u8; 32],
        height: u64,
        attestations: usize,
    },

    /// Payout batch created
    PayoutBatchCreated {
        batch_id: [u8; 32],
        total_reward: u64,
        payout_count: usize,
    },

    /// Payout approved with threshold signature
    PayoutApproved {
        batch_id: [u8; 32],
        signers: usize,
    },

    /// New pool node discovered
    NodeDiscovered {
        peer_id: PeerIdBytes,
        stratum_addr: String,
    },

    /// PPLNS state synchronized
    PPLNSSynced {
        state_hash: PPLNSStateHash,
        worker_count: usize,
        share_count: usize,
    },

    /// Error occurred
    Error { message: String },
}

/// Messages to send to the P2P network
#[derive(Debug, Clone)]
pub enum OutboundMessage {
    /// Broadcast to a topic
    Broadcast { topic: String, message: PoolMessage },

    /// Send to specific peer
    Direct {
        peer_id: PeerIdBytes,
        message: PoolMessage,
    },

    /// Store in DHT
    DhtPut { key: Vec<u8>, value: Vec<u8> },

    /// Get from DHT
    DhtGet { key: Vec<u8> },
}

/// The distributed pool coordinator
pub struct DistributedPoolCoordinator {
    /// Configuration
    config: CoordinatorConfig,

    /// Gossipsub topics
    topics: PoolTopics,

    /// CRDT PPLNS state
    pplns: Arc<RwLock<DistributedPPLNS>>,

    /// Block consensus manager
    block_consensus: Arc<RwLock<BlockConsensusManager>>,

    /// Payout consensus manager
    payout_consensus: Arc<RwLock<PayoutConsensusManager>>,

    /// Pool node discovery
    discovery: Arc<RwLock<PoolNodeDiscovery>>,

    /// Current block height
    current_height: Arc<RwLock<u64>>,

    /// Current block template
    current_template: Arc<RwLock<Option<BlockTemplateMessage>>>,

    /// Our node info
    our_info: Arc<RwLock<PoolNodeInfo>>,

    /// Last heartbeat sent
    last_heartbeat: Arc<RwLock<Instant>>,

    /// Last PPLNS sync
    last_pplns_sync: Arc<RwLock<Instant>>,

    /// Event sender
    event_tx: mpsc::Sender<CoordinatorEvent>,

    /// Outbound message sender
    outbound_tx: mpsc::Sender<OutboundMessage>,

    /// Metrics
    metrics: Arc<RwLock<CoordinatorMetrics>>,
}

/// Coordinator metrics
#[derive(Debug, Default)]
pub struct CoordinatorMetrics {
    /// Shares received
    pub shares_received: u64,

    /// Shares propagated
    pub shares_propagated: u64,

    /// Blocks found
    pub blocks_found: u64,

    /// Blocks confirmed
    pub blocks_confirmed: u64,

    /// Payouts created
    pub payouts_created: u64,

    /// Payouts approved
    pub payouts_approved: u64,

    /// PPLNS syncs
    pub pplns_syncs: u64,

    /// Nodes discovered
    pub nodes_discovered: u64,
}

impl DistributedPoolCoordinator {
    /// Create new coordinator
    pub fn new(
        config: CoordinatorConfig,
        event_tx: mpsc::Sender<CoordinatorEvent>,
        outbound_tx: mpsc::Sender<OutboundMessage>,
    ) -> Self {
        let topics = PoolTopics::new(&config.network_id);

        let our_info = PoolNodeInfo::new(
            config.peer_id,
            config.stratum_port,
            vec![], // Will be filled with multiaddrs
            config.region.clone(),
        );

        Self {
            pplns: Arc::new(RwLock::new(DistributedPPLNS::new(2.0))),
            block_consensus: Arc::new(RwLock::new(BlockConsensusManager::new(config.peer_id))),
            payout_consensus: Arc::new(RwLock::new(PayoutConsensusManager::new(
                config.peer_id,
                config.signer_index,
                config.secret_share,
            ))),
            discovery: Arc::new(RwLock::new(PoolNodeDiscovery::new(&config.network_id))),
            current_height: Arc::new(RwLock::new(0)),
            current_template: Arc::new(RwLock::new(None)),
            our_info: Arc::new(RwLock::new(our_info)),
            last_heartbeat: Arc::new(RwLock::new(Instant::now())),
            last_pplns_sync: Arc::new(RwLock::new(Instant::now())),
            metrics: Arc::new(RwLock::new(CoordinatorMetrics::default())),
            config,
            topics,
            event_tx,
            outbound_tx,
        }
    }

    /// Get the gossipsub topics
    pub fn topics(&self) -> &PoolTopics {
        &self.topics
    }

    /// Handle incoming pool message
    pub async fn handle_message(
        &self,
        from: PeerIdBytes,
        message: PoolMessage,
    ) -> DistributedResult<()> {
        match message {
            PoolMessage::Share(share) => self.handle_share(from, share).await,
            PoolMessage::BlockFound(announcement) => {
                self.handle_block_found(from, announcement).await
            }
            PoolMessage::PPLNSSync(sync) => self.handle_pplns_sync(from, sync).await,
            PoolMessage::Payout(batch) => self.handle_payout_batch(from, batch).await,
            PoolMessage::Heartbeat(heartbeat) => self.handle_heartbeat(from, heartbeat).await,
            PoolMessage::BlockTemplate(template) => {
                self.handle_block_template(from, template).await
            }
            PoolMessage::RequestPPLNSState { from_round } => {
                self.handle_pplns_request(from, from_round).await
            }
            PoolMessage::Attestation(attestation) => {
                self.handle_attestation(from, attestation).await
            }
            PoolMessage::PayoutVoteMsg(vote) => self.handle_payout_vote(from, vote).await,
        }
    }

    /// Handle incoming share
    async fn handle_share(
        &self,
        from: PeerIdBytes,
        share: DistributedShare,
    ) -> DistributedResult<()> {
        // Verify share
        share.verify()?;

        // Check for duplicate
        let mut pplns = self.pplns.write().await;
        if pplns.has_share(&share.share_id) {
            return Ok(()); // Already have it
        }

        // Add to PPLNS
        pplns.add_share(from, &share);

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.shares_received += 1;
        }

        // Emit event
        let _ = self
            .event_tx
            .send(CoordinatorEvent::ShareReceived {
                share_id: share.share_id,
                worker: share.wallet_address().to_string(),
                difficulty: share.difficulty,
            })
            .await;

        // Propagate to network (if not from us)
        if from != self.config.peer_id {
            let _ = self
                .outbound_tx
                .send(OutboundMessage::Broadcast {
                    topic: self.topics.shares.clone(),
                    message: PoolMessage::Share(share),
                })
                .await;

            let mut metrics = self.metrics.write().await;
            metrics.shares_propagated += 1;
        }

        Ok(())
    }

    /// Handle block found announcement
    async fn handle_block_found(
        &self,
        from: PeerIdBytes,
        announcement: BlockFoundAnnouncement,
    ) -> DistributedResult<()> {
        info!(
            "📦 Block found announcement: height={}, finder={}",
            announcement.height, announcement.finder_wallet
        );

        // Register with consensus manager
        {
            let mut consensus = self.block_consensus.write().await;
            consensus.handle_announcement(announcement.clone())?;
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.blocks_found += 1;
        }

        // Emit event
        let _ = self
            .event_tx
            .send(CoordinatorEvent::BlockFound {
                block_hash: announcement.block_hash,
                height: announcement.height,
                finder: announcement.finder_wallet.clone(),
            })
            .await;

        // Validate and create our attestation
        let pplns_hash = {
            let mut pplns = self.pplns.write().await;
            pplns.state_hash()
        };

        let attestation = NodeAttestation::new(
            announcement.block_hash,
            announcement.height,
            self.config.peer_id,
            true, // Assume valid for now - in production, verify block
            None,
            pplns_hash,
        );

        // Broadcast attestation
        let _ = self
            .outbound_tx
            .send(OutboundMessage::Broadcast {
                topic: self.topics.attestations.clone(),
                message: PoolMessage::Attestation(attestation),
            })
            .await;

        // Propagate announcement if not from us
        if from != self.config.peer_id {
            let _ = self
                .outbound_tx
                .send(OutboundMessage::Broadcast {
                    topic: self.topics.blocks_found.clone(),
                    message: PoolMessage::BlockFound(announcement),
                })
                .await;
        }

        Ok(())
    }

    /// Handle attestation
    async fn handle_attestation(
        &self,
        _from: PeerIdBytes,
        attestation: NodeAttestation,
    ) -> DistributedResult<()> {
        let status = {
            let mut consensus = self.block_consensus.write().await;
            consensus.handle_attestation(attestation.clone())?
        };

        if status == ConsensusStatus::Confirmed {
            info!(
                "✅ Block confirmed: height={}, hash={}",
                attestation.height,
                hex::encode(&attestation.block_hash[..8])
            );

            // Update metrics
            {
                let mut metrics = self.metrics.write().await;
                metrics.blocks_confirmed += 1;
            }

            // Get attestation count
            let attestation_count = {
                let consensus = self.block_consensus.read().await;
                consensus
                    .get_status(&attestation.block_hash)
                    .map(|s| s.attestation_count())
                    .unwrap_or(0)
            };

            // Emit event
            let _ = self
                .event_tx
                .send(CoordinatorEvent::BlockConfirmed {
                    block_hash: attestation.block_hash,
                    height: attestation.height,
                    attestations: attestation_count,
                })
                .await;

            // Trigger payout creation
            self.create_payout_for_block(attestation.block_hash, attestation.height)
                .await?;
        }

        Ok(())
    }

    /// Handle PPLNS sync message
    async fn handle_pplns_sync(
        &self,
        from: PeerIdBytes,
        sync: PPLNSSyncMessage,
    ) -> DistributedResult<()> {
        debug!("📊 PPLNS sync from {:?}", &from[..4]);

        // Merge with our state
        let (merged_hash, worker_count, share_count) = {
            let mut pplns = self.pplns.write().await;

            // Only merge if their state is newer or different
            let our_hash = pplns.state_hash();
            if our_hash != sync.state_hash {
                pplns.merge(&sync.state);
            }

            let new_hash = pplns.state_hash();
            (new_hash, pplns.worker_count(), pplns.share_count())
        };

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.pplns_syncs += 1;
        }

        // Emit event
        let _ = self
            .event_tx
            .send(CoordinatorEvent::PPLNSSynced {
                state_hash: merged_hash,
                worker_count,
                share_count,
            })
            .await;

        Ok(())
    }

    /// Handle PPLNS state request
    async fn handle_pplns_request(
        &self,
        from: PeerIdBytes,
        _from_round: u64,
    ) -> DistributedResult<()> {
        // Send our current state
        let sync_msg = {
            let mut pplns = self.pplns.write().await;
            PPLNSSyncMessage {
                sender: self.config.peer_id,
                state: pplns.clone(),
                state_hash: pplns.state_hash(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            }
        };

        let _ = self
            .outbound_tx
            .send(OutboundMessage::Direct {
                peer_id: from,
                message: PoolMessage::PPLNSSync(sync_msg),
            })
            .await;

        Ok(())
    }

    /// Handle payout batch
    async fn handle_payout_batch(
        &self,
        _from: PeerIdBytes,
        batch: PayoutBatch,
    ) -> DistributedResult<()> {
        info!(
            "💰 Payout batch received: {} payouts, total={}",
            batch.payout_count(),
            batch.total_reward
        );

        // Register batch
        {
            let mut payout = self.payout_consensus.write().await;
            payout.handle_batch(batch.clone())?;
        }

        // Emit event
        let _ = self
            .event_tx
            .send(CoordinatorEvent::PayoutBatchCreated {
                batch_id: batch.batch_id,
                total_reward: batch.total_reward,
                payout_count: batch.payout_count(),
            })
            .await;

        // Create and broadcast our vote
        let our_pplns_hash = {
            let mut pplns = self.pplns.write().await;
            pplns.state_hash()
        };

        let vote = {
            let payout = self.payout_consensus.read().await;

            // Verify PPLNS hash matches
            let approve = batch.pplns_state_hash == our_pplns_hash;
            let reason = if !approve {
                Some("PPLNS state mismatch".to_string())
            } else {
                None
            };

            payout.create_vote(batch.batch_id, approve, reason, our_pplns_hash)?
        };

        // Broadcast vote
        let _ = self
            .outbound_tx
            .send(OutboundMessage::Broadcast {
                topic: self.topics.payout_votes.clone(),
                message: PoolMessage::PayoutVoteMsg(vote),
            })
            .await;

        Ok(())
    }

    /// Handle payout vote
    async fn handle_payout_vote(
        &self,
        _from: PeerIdBytes,
        vote: PayoutVote,
    ) -> DistributedResult<()> {
        let status = {
            let mut payout = self.payout_consensus.write().await;
            payout.handle_vote(vote.clone())?
        };

        if status == PayoutStatus::Approved {
            info!("✅ Payout approved: {}", hex::encode(&vote.batch_id[..8]));

            // Update metrics
            {
                let mut metrics = self.metrics.write().await;
                metrics.payouts_approved += 1;
            }

            // Get signer count
            let signer_count = {
                let payout = self.payout_consensus.read().await;
                payout
                    .get_pending(&vote.batch_id)
                    .map(|s| s.approval_count())
                    .unwrap_or(0)
            };

            // Emit event
            let _ = self
                .event_tx
                .send(CoordinatorEvent::PayoutApproved {
                    batch_id: vote.batch_id,
                    signers: signer_count,
                })
                .await;
        }

        Ok(())
    }

    /// Handle heartbeat
    async fn handle_heartbeat(
        &self,
        _from: PeerIdBytes,
        heartbeat: PoolHeartbeat,
    ) -> DistributedResult<()> {
        // Update discovery with node info
        let node_info = PoolNodeInfo {
            peer_id: heartbeat.peer_id,
            stratum_port: heartbeat.stratum_port,
            multiaddrs: vec![], // Would be filled from libp2p
            worker_count: heartbeat.worker_count,
            hashrate: heartbeat.hashrate,
            uptime_seconds: 0,
            region: String::new(),
            version: String::new(),
            last_seen: heartbeat.timestamp / 1000,
            pplns_state_hash: heartbeat.pplns_state_hash,
            accepting_connections: true,
            signature: heartbeat.signature,
        };

        {
            let mut discovery = self.discovery.write().await;
            discovery.add_node(node_info);
        }

        // Register with consensus managers
        {
            let mut block_consensus = self.block_consensus.write().await;
            block_consensus.add_known_node(heartbeat.peer_id);
        }
        {
            let mut payout_consensus = self.payout_consensus.write().await;
            payout_consensus.add_known_node(heartbeat.peer_id);
        }

        Ok(())
    }

    /// Handle block template
    async fn handle_block_template(
        &self,
        _from: PeerIdBytes,
        template: BlockTemplateMessage,
    ) -> DistributedResult<()> {
        // Update current template
        {
            let mut current = self.current_template.write().await;
            *current = Some(template.clone());
        }

        // Update height
        {
            let mut height = self.current_height.write().await;
            *height = template.height;
        }

        Ok(())
    }

    /// Create payout for confirmed block
    async fn create_payout_for_block(
        &self,
        block_hash: [u8; 32],
        height: u64,
    ) -> DistributedResult<()> {
        // Get rewards from PPLNS
        let (rewards, pplns_hash, total_reward) = {
            let mut pplns = self.pplns.write().await;

            // Assume 1 QNK block reward for now
            let block_reward = 1_000_000_000u64;
            let rewards = pplns.calculate_rewards(block_reward);
            let hash = pplns.state_hash();

            (rewards, hash, block_reward)
        };

        if rewards.is_empty() {
            warn!("No rewards to distribute for block {}", height);
            return Ok(());
        }

        // Create payout batch
        let payouts: Vec<_> = rewards.into_iter().map(|r| r.into()).collect();
        let batch = PayoutBatch::new(
            height,
            block_hash,
            total_reward,
            payouts,
            pplns_hash,
            self.config.peer_id,
        );

        // Register and broadcast
        {
            let mut payout = self.payout_consensus.write().await;
            payout.handle_batch(batch.clone())?;
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.payouts_created += 1;
        }

        // Broadcast batch
        let _ = self
            .outbound_tx
            .send(OutboundMessage::Broadcast {
                topic: self.topics.payouts.clone(),
                message: PoolMessage::Payout(batch.clone()),
            })
            .await;

        // Emit event
        let _ = self
            .event_tx
            .send(CoordinatorEvent::PayoutBatchCreated {
                batch_id: batch.batch_id,
                total_reward: batch.total_reward,
                payout_count: batch.payout_count(),
            })
            .await;

        Ok(())
    }

    /// Submit a new share (from local Stratum server)
    pub async fn submit_share(&self, share: DistributedShare) -> DistributedResult<()> {
        // Verify share
        share.verify()?;

        // Add to PPLNS
        {
            let mut pplns = self.pplns.write().await;
            pplns.add_share(self.config.peer_id, &share);
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.shares_received += 1;
        }

        // Broadcast to network
        let _ = self
            .outbound_tx
            .send(OutboundMessage::Broadcast {
                topic: self.topics.shares.clone(),
                message: PoolMessage::Share(share.clone()),
            })
            .await;

        // Emit event
        let _ = self
            .event_tx
            .send(CoordinatorEvent::ShareReceived {
                share_id: share.share_id,
                worker: share.wallet_address().to_string(),
                difficulty: share.difficulty,
            })
            .await;

        Ok(())
    }

    /// Announce block found (from local Stratum server)
    pub async fn announce_block_found(
        &self,
        announcement: BlockFoundAnnouncement,
    ) -> DistributedResult<()> {
        self.handle_block_found(self.config.peer_id, announcement)
            .await
    }

    /// Periodic maintenance tick
    pub async fn tick(&self) -> DistributedResult<()> {
        // Check if we need to send heartbeat
        let should_heartbeat = {
            let last = self.last_heartbeat.read().await;
            last.elapsed() >= self.config.heartbeat_interval
        };

        if should_heartbeat {
            self.send_heartbeat().await?;
            *self.last_heartbeat.write().await = Instant::now();
        }

        // Check if we need to sync PPLNS
        let should_sync = {
            let last = self.last_pplns_sync.read().await;
            last.elapsed() >= self.config.pplns_sync_interval
        };

        if should_sync {
            self.broadcast_pplns_state().await?;
            *self.last_pplns_sync.write().await = Instant::now();
        }

        // Cleanup stale nodes
        {
            let mut discovery = self.discovery.write().await;
            discovery.remove_stale_nodes();
        }

        Ok(())
    }

    /// Send heartbeat to network
    async fn send_heartbeat(&self) -> DistributedResult<()> {
        let pplns_hash = {
            let mut pplns = self.pplns.write().await;
            pplns.state_hash()
        };

        let our_info = self.our_info.read().await;
        let height = *self.current_height.read().await;

        let heartbeat = PoolHeartbeat {
            peer_id: self.config.peer_id,
            stratum_port: self.config.stratum_port,
            worker_count: our_info.worker_count,
            hashrate: our_info.hashrate,
            block_height: height,
            pplns_state_hash: pplns_hash,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            signature: [0u8; 64], // Would be signed
        };

        let _ = self
            .outbound_tx
            .send(OutboundMessage::Broadcast {
                topic: self.topics.heartbeat.clone(),
                message: PoolMessage::Heartbeat(heartbeat),
            })
            .await;

        Ok(())
    }

    /// Broadcast PPLNS state for synchronization
    async fn broadcast_pplns_state(&self) -> DistributedResult<()> {
        let sync_msg = {
            let mut pplns = self.pplns.write().await;
            PPLNSSyncMessage {
                sender: self.config.peer_id,
                state: pplns.clone(),
                state_hash: pplns.state_hash(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            }
        };

        let _ = self
            .outbound_tx
            .send(OutboundMessage::Broadcast {
                topic: self.topics.pplns_state.clone(),
                message: PoolMessage::PPLNSSync(sync_msg),
            })
            .await;

        Ok(())
    }

    /// Update our node info
    pub async fn update_node_info(&self, worker_count: u32, hashrate: f64) {
        let mut info = self.our_info.write().await;
        info.worker_count = worker_count;
        info.hashrate = hashrate;
        info.last_seen = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    /// Get current PPLNS state hash
    pub async fn pplns_state_hash(&self) -> PPLNSStateHash {
        let mut pplns = self.pplns.write().await;
        pplns.state_hash()
    }

    /// Get worker statistics (wallet, difficulty, proportion)
    pub async fn worker_stats(&self) -> Vec<(String, f64, f64)> {
        let pplns = self.pplns.read().await;
        pplns.worker_stats()
    }

    /// v10.0.0: Get distributed PPLNS proportions for block producer coinbase distribution.
    /// Returns raw proportions (summing to 1.0) — NO fee deduction here.
    /// Block producer's existing fee logic handles dev/pool fees.
    /// Returns None if no shares in the CRDT window.
    pub async fn get_distributed_pplns_proportions(&self) -> Option<Vec<(String, f64)>> {
        let stats = self.worker_stats().await;
        if stats.is_empty() {
            return None;
        }
        // worker_stats returns (wallet, difficulty, proportion) — extract (wallet, proportion)
        let proportions: Vec<(String, f64)> = stats
            .into_iter()
            .filter(|(_, _, proportion)| *proportion > 0.0)
            .map(|(wallet, _, proportion)| (wallet, proportion))
            .collect();

        if proportions.is_empty() {
            None
        } else {
            Some(proportions)
        }
    }

    /// Get known pool nodes
    pub async fn known_nodes(&self) -> Vec<PoolNodeInfo> {
        let discovery = self.discovery.read().await;
        discovery.get_all_nodes().into_iter().cloned().collect()
    }

    /// Get metrics
    pub async fn metrics(&self) -> CoordinatorMetrics {
        let metrics = self.metrics.read().await;
        CoordinatorMetrics {
            shares_received: metrics.shares_received,
            shares_propagated: metrics.shares_propagated,
            blocks_found: metrics.blocks_found,
            blocks_confirmed: metrics.blocks_confirmed,
            payouts_created: metrics.payouts_created,
            payouts_approved: metrics.payouts_approved,
            pplns_syncs: metrics.pplns_syncs,
            nodes_discovered: metrics.nodes_discovered,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coordinator_creation() {
        let (event_tx, _event_rx) = mpsc::channel(100);
        let (outbound_tx, _outbound_rx) = mpsc::channel(100);

        let config = CoordinatorConfig {
            peer_id: [1u8; 32],
            network_id: "testnet".to_string(),
            ..Default::default()
        };

        let coordinator = DistributedPoolCoordinator::new(config, event_tx, outbound_tx);

        assert_eq!(coordinator.topics().network_id, "testnet");
    }

    #[tokio::test]
    async fn test_pplns_state_hash() {
        let (event_tx, _event_rx) = mpsc::channel(100);
        let (outbound_tx, _outbound_rx) = mpsc::channel(100);

        let config = CoordinatorConfig::default();
        let coordinator = DistributedPoolCoordinator::new(config, event_tx, outbound_tx);

        let hash = coordinator.pplns_state_hash().await;
        assert_ne!(hash, [0u8; 32]); // Should have some hash
    }
}
