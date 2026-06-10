/// Persistent Peer Session Management for Q-NarwhalKnight
///
/// Implements long-lived peer connections for blockchain state synchronization
/// and consensus participation across the anonymous mesh network.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

use crate::connection_manager::{ActiveConnection, ConnectionManager};
use crate::peer_registry::PeerInfo;
use q_types::{Block, Transaction, NodeId, ProposalId, Vote};

/// Message types for peer-to-peer communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PeerMessage {
    /// Handshake and session establishment
    Handshake {
        node_id: String,
        protocol_version: String,
        capabilities: Vec<String>,
        timestamp: u64,
    },
    HandshakeResponse {
        node_id: String,
        accepted: bool,
        message: String,
    },

    /// Keep-alive for persistent connections
    KeepAlive {
        timestamp: u64,
        last_activity: u64,
    },
    KeepAliveResponse {
        timestamp: u64,
        status: String,
    },

    /// Blockchain synchronization
    StateRequest {
        from_height: u64,
        to_height: Option<u64>,
        request_id: String,
    },
    StateResponse {
        blocks: Vec<Block>,
        current_height: u64,
        request_id: String,
        has_more: bool,
    },

    /// Consensus participation
    BlockProposal {
        proposal_id: ProposalId,
        block: Block,
        proposer: NodeId,
        epoch: u64,
    },
    Vote {
        vote: Vote,
        voter: NodeId,
        epoch: u64,
    },
    ConsensusAck {
        proposal_id: ProposalId,
        ack_type: String,
    },

    /// Transaction propagation
    TransactionBroadcast {
        transactions: Vec<Transaction>,
        source: NodeId,
    },
    TransactionAck {
        tx_hashes: Vec<String>,
        status: String,
    },
}

/// Persistent peer session state
#[derive(Debug)]
pub struct PeerSession {
    pub peer_info: PeerInfo,
    pub stream: Arc<Mutex<TcpStream>>,
    pub established_at: SystemTime,
    pub last_activity: SystemTime,
    pub last_keep_alive: SystemTime,
    pub session_id: String,
    pub protocol_version: String,
    pub capabilities: Vec<String>,
    pub message_count: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub is_syncing: bool,
    pub last_sync_height: u64,
}

/// Persistent peer manager
#[derive(Clone)]
pub struct PersistentPeerManager {
    sessions: Arc<RwLock<HashMap<String, PeerSession>>>,
    connection_manager: Arc<ConnectionManager>,
    message_sender: Arc<Mutex<Option<mpsc::UnboundedSender<PeerMessage>>>>,
    keep_alive_interval: Duration,
    session_timeout: Duration,
    max_sessions: usize,
}

impl PersistentPeerManager {
    /// Create new persistent peer manager
    pub fn new(connection_manager: Arc<ConnectionManager>) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            connection_manager,
            message_sender: Arc::new(Mutex::new(None)),
            keep_alive_interval: Duration::from_secs(30),
            session_timeout: Duration::from_secs(300), // 5 minutes
            max_sessions: 50,
        }
    }

    /// Start persistent peer management
    pub async fn start(&self) -> Result<mpsc::UnboundedReceiver<PeerMessage>> {
        let (tx, rx) = mpsc::unbounded_channel();
        *self.message_sender.lock().await = Some(tx);

        // Start background tasks
        self.start_keep_alive_task().await;
        self.start_session_cleanup_task().await;
        self.start_peer_monitoring_task().await;

        info!("🔗 Persistent peer manager started");
        Ok(rx)
    }

    /// Establish persistent session with a peer
    pub async fn establish_session(&self, peer_info: PeerInfo, stream: TcpStream) -> Result<String> {
        // Convert validator_id ([u8; 32]) to hex string for session ID
        let validator_hex = hex::encode(peer_info.validator_id);
        let session_id = format!("{}_{}", validator_hex,
                                SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs());

        info!("🤝 Establishing persistent session with {}: {}", validator_hex, session_id);

        // Send handshake
        let handshake = PeerMessage::Handshake {
            node_id: "q-narwhalknight-beta".to_string(),
            protocol_version: "1.0.0".to_string(),
            capabilities: vec![
                "consensus".to_string(),
                "state_sync".to_string(),
                "transaction_relay".to_string(),
                "anonymous_messaging".to_string(),
            ],
            timestamp: SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs(),
        };

        let stream = Arc::new(Mutex::new(stream));
        self.send_message(&stream, &handshake).await?;

        // Wait for handshake response
        let response = self.receive_message(&stream).await?;

        match response {
            PeerMessage::HandshakeResponse { accepted: true, node_id, .. } => {
                info!("✅ Handshake accepted by peer: {}", node_id);

                let session = PeerSession {
                    peer_info: peer_info.clone(),
                    stream: stream.clone(),
                    established_at: SystemTime::now(),
                    last_activity: SystemTime::now(),
                    last_keep_alive: SystemTime::now(),
                    session_id: session_id.clone(),
                    protocol_version: "1.0.0".to_string(),
                    capabilities: vec![
                        "consensus".to_string(),
                        "state_sync".to_string(),
                        "transaction_relay".to_string(),
                    ],
                    message_count: 1,
                    bytes_sent: 0,
                    bytes_received: 0,
                    is_syncing: false,
                    last_sync_height: 0,
                };

                let mut sessions = self.sessions.write().await;
                sessions.insert(session_id.clone(), session);

                info!("🎉 Persistent session established: {} (total: {})", session_id, sessions.len());

                // Start message handling for this session
                self.start_session_handler(session_id.clone(), stream).await;

                Ok(session_id)
            }
            PeerMessage::HandshakeResponse { accepted: false, message, .. } => {
                warn!("❌ Handshake rejected: {}", message);
                Err(anyhow!("Handshake rejected: {}", message))
            }
            _ => {
                warn!("❌ Unexpected response to handshake");
                Err(anyhow!("Unexpected handshake response"))
            }
        }
    }

    /// Send keep-alive to all sessions
    async fn send_keep_alives(&self) -> Result<()> {
        let sessions = self.sessions.read().await;
        let now = SystemTime::now();

        for (session_id, session) in sessions.iter() {
            if now.duration_since(session.last_keep_alive).unwrap_or_default() >= self.keep_alive_interval {
                let keep_alive = PeerMessage::KeepAlive {
                    timestamp: now.duration_since(SystemTime::UNIX_EPOCH)?.as_secs(),
                    last_activity: session.last_activity.duration_since(SystemTime::UNIX_EPOCH)?.as_secs(),
                };

                match self.send_message(&session.stream, &keep_alive).await {
                    Ok(()) => {
                        debug!("💓 Keep-alive sent to session: {}", session_id);
                    }
                    Err(e) => {
                        warn!("❌ Failed to send keep-alive to {}: {}", session_id, e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Send blockchain state request to peers
    pub async fn request_blockchain_state(&self, from_height: u64, to_height: Option<u64>) -> Result<()> {
        let sessions = self.sessions.read().await;

        if sessions.is_empty() {
            warn!("⚠️ No persistent sessions available for state sync");
            return Ok(());
        }

        let request_id = format!("state_req_{}",
                                SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs());

        let state_request = PeerMessage::StateRequest {
            from_height,
            to_height,
            request_id: request_id.clone(),
        };

        // Send to the first available session (could be improved with peer selection)
        if let Some((session_id, session)) = sessions.iter().next() {
            info!("📡 Requesting blockchain state from {} to {:?} via session: {}",
                  from_height, to_height, session_id);

            self.send_message(&session.stream, &state_request).await?;
        }

        Ok(())
    }

    /// Broadcast block proposal to all peers
    pub async fn broadcast_block_proposal(&self, proposal_id: ProposalId, block: Block, proposer: NodeId, epoch: u64) -> Result<()> {
        let sessions = self.sessions.read().await;

        if sessions.is_empty() {
            warn!("⚠️ No persistent sessions available for consensus");
            return Ok(());
        }

        let proposal = PeerMessage::BlockProposal {
            proposal_id,
            block,
            proposer,
            epoch,
        };

        info!("📢 Broadcasting block proposal to {} peers", sessions.len());

        for (session_id, session) in sessions.iter() {
            match self.send_message(&session.stream, &proposal).await {
                Ok(()) => {
                    debug!("✅ Block proposal sent to session: {}", session_id);
                }
                Err(e) => {
                    warn!("❌ Failed to send block proposal to {}: {}", session_id, e);
                }
            }
        }

        Ok(())
    }

    /// Broadcast vote to all peers
    pub async fn broadcast_vote(&self, vote: Vote, voter: NodeId, epoch: u64) -> Result<()> {
        let sessions = self.sessions.read().await;

        if sessions.is_empty() {
            warn!("⚠️ No persistent sessions available for voting");
            return Ok(());
        }

        let vote_msg = PeerMessage::Vote {
            vote,
            voter,
            epoch,
        };

        info!("🗳️ Broadcasting vote to {} peers", sessions.len());

        for (session_id, session) in sessions.iter() {
            match self.send_message(&session.stream, &vote_msg).await {
                Ok(()) => {
                    debug!("✅ Vote sent to session: {}", session_id);
                }
                Err(e) => {
                    warn!("❌ Failed to send vote to {}: {}", session_id, e);
                }
            }
        }

        Ok(())
    }

    /// Get session statistics
    pub async fn get_session_stats(&self) -> HashMap<String, serde_json::Value> {
        let sessions = self.sessions.read().await;
        let mut stats = HashMap::new();

        stats.insert("total_sessions".to_string(), serde_json::Value::Number(sessions.len().into()));

        let active_sessions = sessions.iter()
            .filter(|(_, session)| {
                SystemTime::now().duration_since(session.last_activity).unwrap_or_default() < self.session_timeout
            })
            .count();

        stats.insert("active_sessions".to_string(), serde_json::Value::Number(active_sessions.into()));

        let total_messages: u64 = sessions.values().map(|s| s.message_count).sum();
        stats.insert("total_messages".to_string(), serde_json::Value::Number(total_messages.into()));

        let syncing_sessions = sessions.iter()
            .filter(|(_, session)| session.is_syncing)
            .count();

        stats.insert("syncing_sessions".to_string(), serde_json::Value::Number(syncing_sessions.into()));

        stats
    }

    /// Send message to peer
    async fn send_message(&self, stream: &Arc<Mutex<TcpStream>>, message: &PeerMessage) -> Result<()> {
        let message_json = serde_json::to_string(message)?;
        let message_bytes = format!("{}\n", message_json).into_bytes();

        let mut stream = stream.lock().await;
        timeout(Duration::from_secs(10), async {
            stream.write_all(&message_bytes).await?;
            stream.flush().await?;
            Ok::<(), anyhow::Error>(())
        }).await??;

        debug!("📤 Message sent: {} bytes", message_bytes.len());
        Ok(())
    }

    /// Receive message from peer
    async fn receive_message(&self, stream: &Arc<Mutex<TcpStream>>) -> Result<PeerMessage> {
        let mut stream = stream.lock().await;
        let mut buffer = vec![0u8; 8192];

        let bytes_read = timeout(Duration::from_secs(30), stream.read(&mut buffer)).await??;

        if bytes_read == 0 {
            return Err(anyhow!("Connection closed"));
        }

        let message_str = String::from_utf8_lossy(&buffer[..bytes_read]);
        let message: PeerMessage = serde_json::from_str(message_str.trim())?;

        debug!("📥 Message received: {} bytes", bytes_read);
        Ok(message)
    }

    /// Start keep-alive background task
    async fn start_keep_alive_task(&self) {
        let self_clone = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(self_clone.keep_alive_interval);

            loop {
                interval.tick().await;

                if let Err(e) = self_clone.send_keep_alives().await {
                    error!("❌ Keep-alive task error: {}", e);
                }
            }
        });

        info!("💓 Keep-alive task started");
    }

    /// Start session cleanup background task
    async fn start_session_cleanup_task(&self) {
        let self_clone = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));

            loop {
                interval.tick().await;

                let mut sessions = self_clone.sessions.write().await;
                let now = SystemTime::now();
                let mut expired_sessions = Vec::new();

                for (session_id, session) in sessions.iter() {
                    if now.duration_since(session.last_activity).unwrap_or_default() > self_clone.session_timeout {
                        expired_sessions.push(session_id.clone());
                    }
                }

                for session_id in expired_sessions {
                    sessions.remove(&session_id);
                    info!("🗑️ Removed expired session: {}", session_id);
                }
            }
        });

        info!("🧹 Session cleanup task started");
    }

    /// Start peer monitoring task
    async fn start_peer_monitoring_task(&self) {
        let self_clone = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));

            loop {
                interval.tick().await;

                let sessions = self_clone.sessions.read().await;
                let stats = self_clone.get_session_stats().await;

                if !sessions.is_empty() {
                    info!("📊 Peer sessions: {} total, {} active, {} messages",
                          stats.get("total_sessions").unwrap_or(&serde_json::Value::Number(0.into())),
                          stats.get("active_sessions").unwrap_or(&serde_json::Value::Number(0.into())),
                          stats.get("total_messages").unwrap_or(&serde_json::Value::Number(0.into())));
                }
            }
        });

        info!("📊 Peer monitoring task started");
    }

    /// Start message handler for a specific session
    async fn start_session_handler(&self, session_id: String, stream: Arc<Mutex<TcpStream>>) {
        let self_clone = self.clone();
        let session_id_clone = session_id.clone();

        tokio::spawn(async move {
            info!("🎧 Starting message handler for session: {}", session_id_clone);

            loop {
                match self_clone.receive_message(&stream).await {
                    Ok(message) => {
                        match self_clone.handle_session_message(session_id_clone.clone(), message).await {
                            Ok(()) => {},
                            Err(e) => {
                                error!("❌ Error handling message for session {}: {}", session_id_clone, e);
                            }
                        }
                    }
                    Err(e) => {
                        warn!("❌ Session {} disconnected: {}", session_id_clone, e);
                        break;
                    }
                }
            }

            // Remove session on disconnect
            let mut sessions = self_clone.sessions.write().await;
            sessions.remove(&session_id_clone);
            info!("🔌 Session handler ended: {}", session_id_clone);
        });
    }

    /// Handle message from a session
    async fn handle_session_message(&self, session_id: String, message: PeerMessage) -> Result<()> {
        // Update session activity
        {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(&session_id) {
                session.last_activity = SystemTime::now();
                session.message_count += 1;
            }
        }

        // Forward message to main handler
        if let Some(sender) = self.message_sender.lock().await.as_ref() {
            sender.send(message)?;
        }

        Ok(())
    }
}