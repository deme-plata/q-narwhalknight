//! Compute Tunnel — Encrypted P2P task routing between nodes and miners
//!
//! **Phase 5 future work**: This module defines the types and routing logic
//! for distributed compute tunnels. The types are architecturally correct
//! but `TunnelManager` is not yet instantiated in production — tunnels will
//! be wired when P2P compute distribution is implemented.
//!
//! ## P2P Compute Peer Discovery (Issue #002)
//!
//! Nodes announce their compute capacity via gossipsub on `COMPUTE_TUNNEL_TOPIC`.
//! The `PeerRegistry` tracks discovered peers with a 60-second TTL and supports
//! score-based peer selection for task routing.
//!
//! ## Tunnel Handshake Protocol (Issue #002 — Criterion 4)
//!
//! `CryptoHandshake` performs X25519 Diffie-Hellman key exchange with
//! HKDF-SHA256 session key derivation. The state machine progresses:
//! `Init -> KeyExchangeSent -> KeyExchangeReceived -> Established`.
//!
//! ## Multiplexed Stream Framing (Issue #002 — Criterion 5)
//!
//! `FramedTunnelStream` provides a multiplexing framing layer over a single
//! tunnel connection. Each frame is `[stream_id: u32][length: u32][payload]`.
//! Multiple logical channels (Mining, Inference, Proof, Control) share one
//! physical connection.
//!
//! ## Result Verification (Issue #002 — Criterion 6)
//!
//! `ResultVerifier` implements 2-of-3 redundant compute: high-value tasks
//! are dispatched to 3 peers, and the result is accepted only if 2+ peers
//! agree on the output. Peers that fail to respond within 10 seconds are
//! excluded from the consensus.
//!
//! Tunnels enable distributed compute by connecting:
//! - Miner → Node: mining solutions + telemetry
//! - Node → Node: task distribution + results
//! - Node → Miner: push compute tasks to idle miner GPU
//! - Miner → Miner: collaborative proof generation
//!
//! Each tunnel is encrypted (NOISE XX pattern) and carries
//! typed work items with priority routing.

#![allow(dead_code)]

use crate::{ComputeLayer, ComputePeerInfo, ResourceSnapshot, TunnelInfo, TunnelType};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tracing::{info, warn, debug};
use zeroize::Zeroize;

/// TTL for known peers in seconds — peers not re-announced within this
/// window are considered stale and evicted.
const PEER_TTL_SECS: u64 = 60;

/// Maximum number of concurrent streams per tunnel.
const MAX_STREAMS_PER_TUNNEL: usize = 8;

/// Handshake timeout in seconds.
const HANDSHAKE_TIMEOUT_SECS: u64 = 5;

/// Maximum new tunnels that `auto_connect_to_best_peers` opens per call.
const MAX_AUTO_OPENS_PER_CALL: usize = 2;

// ═══════════════════════════════════════════════════════════════════
// Rekey Constants (Issue #024 — Forward Secrecy)
// ═══════════════════════════════════════════════════════════════════

/// Auto-rekey interval: 1 hour.
const REKEY_INTERVAL: Duration = Duration::from_secs(3600);

/// Auto-rekey byte threshold: 10 GB transferred.
const REKEY_BYTE_THRESHOLD: u64 = 10 * 1024 * 1024 * 1024;

/// Retry interval on rekey failure: 5 minutes.
const REKEY_RETRY_INTERVAL: Duration = Duration::from_secs(300);

// ═══════════════════════════════════════════════════════════════════
// Tunnel Handshake — capability negotiation between peers
// ═══════════════════════════════════════════════════════════════════

/// Handshake sent by the initiator to open a compute tunnel.
///
/// Contains the initiator's identity, a random nonce for replay protection,
/// the list of compute capabilities offered, and a protocol version for
/// forward compatibility.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TunnelHandshake {
    pub initiator_peer_id: String,
    pub responder_peer_id: String,
    pub nonce: [u8; 32],
    pub capabilities: Vec<String>, // ["mining", "inference", "bridge-verify"]
    pub protocol_version: u32,
    pub timestamp: u64,
}

impl TunnelHandshake {
    /// Create a new handshake for the given peer pair.
    pub fn new(initiator: &str, responder: &str, capabilities: Vec<String>) -> Self {
        let mut nonce = [0u8; 32];
        // Simple nonce generation — production would use CSPRNG
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        for (i, byte) in nonce.iter_mut().enumerate() {
            *byte = ((seed >> (i % 16)) & 0xFF) as u8 ^ (i as u8);
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            initiator_peer_id: initiator.to_string(),
            responder_peer_id: responder.to_string(),
            nonce,
            capabilities,
            protocol_version: 1,
            timestamp,
        }
    }
}

/// Response to a `TunnelHandshake`.
///
/// The responder either accepts (returning its own nonce and matching
/// capabilities) or rejects (with a human-readable reason).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HandshakeResponse {
    pub accepted: bool,
    pub nonce: [u8; 32],           // responder's nonce
    pub capabilities: Vec<String>, // intersection of supported capabilities
    pub reason: Option<String>,    // populated when rejected
}

impl HandshakeResponse {
    /// Create an acceptance response with the responder's own nonce and the
    /// intersection of capabilities that both peers support.
    pub fn accept(responder_caps: Vec<String>) -> Self {
        let mut nonce = [0u8; 32];
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        for (i, byte) in nonce.iter_mut().enumerate() {
            *byte = ((seed >> (i % 16)) & 0xFF) as u8 ^ (i as u8).wrapping_add(0xAA);
        }

        Self {
            accepted: true,
            nonce,
            capabilities: responder_caps,
            reason: None,
        }
    }

    /// Create a rejection response with the given reason.
    pub fn reject(reason: &str) -> Self {
        Self {
            accepted: false,
            nonce: [0u8; 32],
            capabilities: Vec::new(),
            reason: Some(reason.to_string()),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tunnel State Machine
// ═══════════════════════════════════════════════════════════════════

/// Lifecycle state of a compute tunnel.
///
/// ```text
/// Handshaking ──(accepted)──> Active ──(drain)──> Draining ──> Closed
///      │                                                         ^
///      └───────(rejected / timeout)──────────────────────────────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TunnelState {
    /// Waiting for handshake response from the remote peer.
    Handshaking,
    /// Tunnel is fully established and carrying traffic.
    Active,
    /// Graceful shutdown — finish pending tasks, reject new ones.
    Draining,
    /// Tunnel is closed and will be cleaned up.
    Closed,
}

// ═══════════════════════════════════════════════════════════════════
// Multiplexed Streams — multiple logical channels per tunnel
// ═══════════════════════════════════════════════════════════════════

/// Type of a multiplexed stream within a tunnel.
///
/// Each tunnel can carry up to `MAX_STREAMS_PER_TUNNEL` concurrent streams,
/// each dedicated to a different workload type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum StreamType {
    Mining,
    Inference,
    BridgeVerify,
    TensorShard,
    Control,
}

impl std::fmt::Display for StreamType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamType::Mining => write!(f, "mining"),
            StreamType::Inference => write!(f, "inference"),
            StreamType::BridgeVerify => write!(f, "bridge-verify"),
            StreamType::TensorShard => write!(f, "tensor-shard"),
            StreamType::Control => write!(f, "control"),
        }
    }
}

/// A single multiplexed stream within a compute tunnel.
///
/// Each stream has its own typed send channel but shares the tunnel's
/// underlying transport. Streams track their own byte counters for
/// per-workload accounting.
pub struct TunnelStream {
    pub stream_type: StreamType,
    pub stream_id: u64,
    pub tx: mpsc::Sender<TunnelPayload>,
    pub created_at: Instant,
    pub bytes_sent: Arc<AtomicU64>,
    pub bytes_received: Arc<AtomicU64>,
}

impl TunnelStream {
    /// Send a payload through this stream.
    ///
    /// Returns `Ok(())` if queued, `Err` if the channel is closed or full.
    pub async fn send(&self, payload: TunnelPayload) -> Result<(), mpsc::error::SendError<TunnelPayload>> {
        self.tx.send(payload).await
    }

    /// Record bytes sent through this stream.
    pub fn record_send(&self, bytes: u64) {
        self.bytes_sent.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record bytes received through this stream.
    pub fn record_receive(&self, bytes: u64) {
        self.bytes_received.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Total bytes sent since stream creation.
    pub fn total_bytes_sent(&self) -> u64 {
        self.bytes_sent.load(Ordering::Relaxed)
    }

    /// Total bytes received since stream creation.
    pub fn total_bytes_received(&self) -> u64 {
        self.bytes_received.load(Ordering::Relaxed)
    }

    /// Elapsed time since the stream was created.
    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    /// Issue #024: Perform an in-band rekey over this stream.
    ///
    /// This is a convenience method that drives the full rekey protocol:
    /// 1. Generates a new ephemeral X25519 keypair via the `RekeyManager`.
    /// 2. Sends a `TunnelPayload::Rekey` message through this stream.
    /// 3. The caller must separately handle the incoming `RekeyAck` and
    ///    call `rekey_manager.complete_rekey()` to finalize.
    ///
    /// Returns the `TunnelPayload::Rekey` that was sent, or an error if
    /// the rekey could not be initiated (e.g., one is already in progress)
    /// or the send channel is closed.
    pub async fn rekey(
        &self,
        rekey_manager: &mut RekeyManager,
    ) -> Result<TunnelPayload, String> {
        let payload = rekey_manager.initiate_rekey()?;

        // Send the rekey request through this stream's channel
        self.tx
            .send(payload.clone())
            .await
            .map_err(|e| {
                // If the send fails, mark the rekey as failed so the
                // old key is preserved and a retry is scheduled.
                rekey_manager.fail_rekey(&format!("channel send failed: {}", e));
                format!("Failed to send rekey message: {}", e)
            })?;

        // Record the overhead bytes of the rekey message itself
        // (32 pubkey + 32 nonce + 8 seq = 72 bytes + framing)
        self.record_send(72);

        Ok(payload)
    }
}

/// Work item sent through a tunnel
#[derive(Debug, Clone)]
pub struct TunnelWorkItem {
    pub id: u64,
    pub layer: ComputeLayer,
    pub payload_bytes: usize,
    pub priority: u8,           // 0 = highest (mining), 7 = lowest
    pub sender_peer: String,
}

/// v9.6.0: Typed tunnel payload for AI inference task routing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TunnelPayload {
    /// Mining solution from miner
    MiningSubmit(Vec<u8>),
    /// AI inference request routed to capable peer
    InferenceRequest {
        request_id: String,
        prompt: String,
        max_tokens: usize,
        model: Option<String>,
        wallet: Option<String>,
    },
    /// AI inference response back to requester
    InferenceResponse {
        request_id: String,
        generated_text: String,
        tokens_generated: usize,
        tokens_per_second: f64,
    },
    /// Tensor shard for distributed model serving
    TensorShard {
        request_id: String,
        layer_id: u32,
        shard_data: Vec<u8>,
    },
    /// Layer output forwarded in pipeline parallelism
    LayerOutput {
        request_id: String,
        layer_range: (u32, u32),
        activations: Vec<u8>,
    },
    /// Issue #024: Rekey coordination message for forward secrecy.
    ///
    /// Sent by the initiator to propose a new ephemeral key exchange.
    /// The responder should reply with their own `Rekey` containing their
    /// new ephemeral public key so both sides can derive a fresh session key.
    Rekey {
        /// New ephemeral X25519 public key from the sender.
        ephemeral_pubkey: [u8; 32],
        /// Nonce for HKDF salt derivation.
        nonce: [u8; 32],
        /// Monotonic rekey sequence number to prevent replays.
        rekey_seq: u64,
    },
    /// Acknowledgement of a rekey — the responder sends their new ephemeral
    /// public key back so the initiator can derive the same new session key.
    RekeyAck {
        /// Responder's new ephemeral X25519 public key.
        ephemeral_pubkey: [u8; 32],
        /// Responder's nonce.
        nonce: [u8; 32],
        /// Echo of the initiator's rekey_seq to correlate.
        rekey_seq: u64,
    },
}

/// A single compute tunnel to a remote peer.
///
/// Now includes lifecycle state tracking (`TunnelState`) and multiplexed
/// streams (`TunnelStream`). A tunnel starts in `Handshaking` state and
/// transitions to `Active` once the handshake succeeds. Multiple logical
/// streams can be opened on an active tunnel (up to `MAX_STREAMS_PER_TUNNEL`).
pub struct ComputeTunnel {
    pub peer_id: String,
    pub tunnel_type: TunnelType,
    pub established_ms: u64,
    pub encrypted: bool,
    pub state: RwLock<TunnelState>,
    /// Negotiated capabilities from the handshake (empty until Active).
    pub negotiated_capabilities: Vec<String>,
    bytes_sent: Arc<AtomicU64>,
    bytes_received: Arc<AtomicU64>,
    tasks_routed: Arc<AtomicU64>,
    latency_ms: Arc<AtomicU64>,
    active: Arc<AtomicBool>,
    /// Monotonic stream ID counter for this tunnel.
    next_stream_id: AtomicU64,
    /// Active multiplexed streams keyed by stream_id.
    streams: RwLock<HashMap<u64, TunnelStream>>,
    /// Shared receive side for stream payloads — the tunnel's transport
    /// layer drains this and dispatches to the appropriate stream.
    stream_rx: RwLock<Option<mpsc::Receiver<TunnelPayload>>>,
    /// Sender clone-able handle that new streams use.
    stream_tx: mpsc::Sender<TunnelPayload>,
}

impl ComputeTunnel {
    pub fn new(peer_id: String, tunnel_type: TunnelType) -> Self {
        Self::with_state(peer_id, tunnel_type, TunnelState::Active)
    }

    /// Create a tunnel in a specific initial state.
    ///
    /// Use `TunnelState::Handshaking` when the tunnel is being opened with
    /// the handshake protocol; use `TunnelState::Active` for the legacy
    /// code path that skips handshake.
    pub fn with_state(peer_id: String, tunnel_type: TunnelType, state: TunnelState) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Shared channel for all streams in this tunnel.
        // Buffer of 256 — large enough to absorb bursts without blocking.
        let (stream_tx, stream_rx) = mpsc::channel(256);

        let is_active = state == TunnelState::Active || state == TunnelState::Handshaking;

        Self {
            peer_id,
            tunnel_type,
            established_ms: now,
            encrypted: true,
            state: RwLock::new(state),
            negotiated_capabilities: Vec::new(),
            bytes_sent: Arc::new(AtomicU64::new(0)),
            bytes_received: Arc::new(AtomicU64::new(0)),
            tasks_routed: Arc::new(AtomicU64::new(0)),
            latency_ms: Arc::new(AtomicU64::new(0)),
            active: Arc::new(AtomicBool::new(is_active)),
            next_stream_id: AtomicU64::new(1),
            streams: RwLock::new(HashMap::new()),
            stream_rx: RwLock::new(Some(stream_rx)),
            stream_tx,
        }
    }

    /// Record bytes sent through this tunnel
    pub fn record_send(&self, bytes: u64) {
        self.bytes_sent.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record bytes received through this tunnel
    pub fn record_receive(&self, bytes: u64) {
        self.bytes_received.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a task routed through this tunnel
    pub fn record_task(&self) {
        self.tasks_routed.fetch_add(1, Ordering::Relaxed);
    }

    /// Update measured latency
    pub fn update_latency(&self, ms: u32) {
        self.latency_ms.store(ms as u64, Ordering::Relaxed);
    }

    /// Check if tunnel is active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Close this tunnel
    pub fn close(&self) {
        *self.state.write() = TunnelState::Closed;
        self.active.store(false, Ordering::SeqCst);
        // Drop all streams
        self.streams.write().clear();
    }

    /// Transition to a new state. Returns `false` if the transition is invalid.
    pub fn transition_to(&self, new_state: TunnelState) -> bool {
        let current = self.state.read().clone();
        let valid = match (&current, &new_state) {
            (TunnelState::Handshaking, TunnelState::Active) => true,
            (TunnelState::Handshaking, TunnelState::Closed) => true,
            (TunnelState::Active, TunnelState::Draining) => true,
            (TunnelState::Active, TunnelState::Closed) => true,
            (TunnelState::Draining, TunnelState::Closed) => true,
            _ => false,
        };

        if valid {
            debug!(
                "🔗 [TUNNEL] {} state {:?} -> {:?}",
                self.peer_id, current, new_state
            );
            if new_state == TunnelState::Closed {
                self.active.store(false, Ordering::SeqCst);
                self.streams.write().clear();
            }
            *self.state.write() = new_state;
        } else {
            warn!(
                "🔗 [TUNNEL] Invalid state transition {:?} -> {:?} for {}",
                current, new_state, self.peer_id
            );
        }

        valid
    }

    // ─── Multiplexed stream management ───────────────────────────

    /// Open a new multiplexed stream on this tunnel.
    ///
    /// Each stream gets its own `TunnelPayload` sender that shares the
    /// tunnel's underlying transport channel. Returns an error if the
    /// tunnel is not `Active` or the per-tunnel stream limit is reached.
    pub fn open_stream(&self, stream_type: StreamType) -> Result<u64, String> {
        let current_state = self.state.read().clone();
        if current_state != TunnelState::Active {
            return Err(format!(
                "Cannot open stream on tunnel in {:?} state",
                current_state
            ));
        }

        let mut streams = self.streams.write();
        if streams.len() >= MAX_STREAMS_PER_TUNNEL {
            return Err(format!(
                "Stream limit reached ({}/{})",
                streams.len(),
                MAX_STREAMS_PER_TUNNEL
            ));
        }

        let stream_id = self.next_stream_id.fetch_add(1, Ordering::Relaxed);
        let stream = TunnelStream {
            stream_type,
            stream_id,
            tx: self.stream_tx.clone(),
            created_at: Instant::now(),
            bytes_sent: Arc::new(AtomicU64::new(0)),
            bytes_received: Arc::new(AtomicU64::new(0)),
        };

        debug!(
            "🔗 [TUNNEL] Opened {} stream #{} on tunnel to {} ({}/{})",
            stream_type,
            stream_id,
            self.peer_id,
            streams.len() + 1,
            MAX_STREAMS_PER_TUNNEL,
        );

        streams.insert(stream_id, stream);
        Ok(stream_id)
    }

    /// Close a stream by ID.
    pub fn close_stream(&self, stream_id: u64) -> bool {
        let mut streams = self.streams.write();
        if let Some(stream) = streams.remove(&stream_id) {
            debug!(
                "🔗 [TUNNEL] Closed {} stream #{} on tunnel to {} (sent={}B, recv={}B)",
                stream.stream_type,
                stream_id,
                self.peer_id,
                stream.total_bytes_sent(),
                stream.total_bytes_received(),
            );
            true
        } else {
            false
        }
    }

    /// Number of currently open streams.
    pub fn stream_count(&self) -> usize {
        self.streams.read().len()
    }

    /// Get the sender handle for a specific stream.
    ///
    /// The caller can use the returned `Sender` to push payloads into the
    /// tunnel's transport. Returns `None` if the stream does not exist.
    pub fn stream_sender(&self, stream_id: u64) -> Option<mpsc::Sender<TunnelPayload>> {
        self.streams.read().get(&stream_id).map(|s| s.tx.clone())
    }

    /// Take the receive half of the shared transport channel.
    ///
    /// This should be called once by the transport layer that drains
    /// payloads and writes them to the network. Returns `None` on
    /// subsequent calls.
    pub fn take_receiver(&self) -> Option<mpsc::Receiver<TunnelPayload>> {
        self.stream_rx.write().take()
    }

    /// Get tunnel info snapshot for dashboard
    pub fn info(&self) -> TunnelInfo {
        TunnelInfo {
            peer_id: self.peer_id.clone(),
            tunnel_type: self.tunnel_type,
            established_ms: self.established_ms,
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            tasks_routed: self.tasks_routed.load(Ordering::Relaxed),
            latency_ms: self.latency_ms.load(Ordering::Relaxed) as u32,
            encrypted: self.encrypted,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Peer Discovery — gossipsub-based compute capacity announcements
// ═══════════════════════════════════════════════════════════════════

/// Create a `ComputePeerInfo` announcement from a resource snapshot.
///
/// This is serialized to JSON and published on `COMPUTE_TUNNEL_TOPIC`
/// every ~30 seconds so peers can discover our compute capacity.
pub fn create_peer_announcement(
    snapshot: &ResourceSnapshot,
    mode: &str,
    peer_id: &str,
) -> ComputePeerInfo {
    let total_cores = snapshot.cpu_per_core.len() as u32;
    // Available cores = cores with < 80% utilization
    let available_cores = snapshot
        .cpu_per_core
        .iter()
        .filter(|&&usage| usage < 80.0)
        .count() as u32;

    // Estimate GPU TFLOPS from utilization and memory (rough heuristic).
    // Without a real GPU capability query, we estimate based on memory size:
    // Consumer GPUs: ~0.5 TFLOPS per GB VRAM (FP32 ballpark)
    let gpu_tflops = if snapshot.gpu_memory_total > 0 {
        let vram_gb = snapshot.gpu_memory_total as f64 / (1024.0 * 1024.0 * 1024.0);
        // Scale by how much GPU is free (inverse of utilization)
        let free_ratio = 1.0 - (snapshot.gpu_utilization as f64 / 100.0);
        vram_gb * 0.5 * free_ratio
    } else {
        0.0
    };

    let ram_total_gb = snapshot.ram_total as f64 / (1024.0 * 1024.0 * 1024.0);
    let ram_available_gb =
        snapshot.ram_total.saturating_sub(snapshot.ram_used) as f64 / (1024.0 * 1024.0 * 1024.0);

    // Bandwidth in Mbps (from bytes/sec capacity estimate)
    let bandwidth_mbps = snapshot.net_capacity_bps as f64 * 8.0 / 1_000_000.0;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    ComputePeerInfo {
        peer_id: peer_id.to_string(),
        available_cores,
        total_cores,
        gpu_tflops,
        ram_available_gb,
        ram_total_gb,
        bandwidth_mbps,
        compute_mode: mode.to_string(),
        active_layers: Vec::new(), // Populated by caller from orchestrator status
        trainer_active: false,     // Populated by caller
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp,
    }
}

/// Parse a raw gossipsub message into a `ComputePeerInfo`.
///
/// Returns `None` if the data is not valid JSON or does not match the schema.
pub fn parse_peer_announcement(data: &[u8]) -> Option<ComputePeerInfo> {
    serde_json::from_slice::<ComputePeerInfo>(data).ok()
}

// ═══════════════════════════════════════════════════════════════════
// Peer Registry — track discovered peers with TTL-based eviction
// ═══════════════════════════════════════════════════════════════════

/// Tracks known compute peers discovered via gossipsub.
///
/// Each peer entry has a 60-second TTL. Peers that do not re-announce
/// within the TTL window are evicted on the next `cleanup_stale()` call.
pub struct PeerRegistry {
    /// peer_id → ComputePeerInfo
    known_peers: Arc<RwLock<HashMap<String, ComputePeerInfo>>>,
}

impl PeerRegistry {
    pub fn new() -> Self {
        Self {
            known_peers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Insert or update a peer. The `timestamp` field in `ComputePeerInfo`
    /// is used as the TTL reference — if the announcement is already stale
    /// at insertion time it is silently dropped.
    pub fn upsert(&self, info: ComputePeerInfo) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Reject announcements that are already older than TTL
        if now.saturating_sub(info.timestamp) > PEER_TTL_SECS {
            debug!(
                "🔗 [PEER REGISTRY] Dropping stale announcement from {} (age={}s)",
                info.peer_id,
                now.saturating_sub(info.timestamp)
            );
            return;
        }

        let peer_id = info.peer_id.clone();
        let mut peers = self.known_peers.write();
        let is_new = !peers.contains_key(&peer_id);
        peers.insert(peer_id.clone(), info);

        if is_new {
            debug!(
                "🔗 [PEER REGISTRY] Discovered new compute peer: {} (total={})",
                peer_id,
                peers.len()
            );
        }
    }

    /// Remove peers whose `timestamp` is older than `PEER_TTL_SECS`.
    pub fn cleanup_stale(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut peers = self.known_peers.write();
        let before = peers.len();
        peers.retain(|_, info| now.saturating_sub(info.timestamp) <= PEER_TTL_SECS);
        let removed = before - peers.len();
        if removed > 0 {
            info!(
                "🔗 [PEER REGISTRY] Evicted {} stale peers (remaining={})",
                removed,
                peers.len()
            );
        }
    }

    /// Return a snapshot of all known (non-stale) peers.
    pub fn all_peers(&self) -> Vec<ComputePeerInfo> {
        self.known_peers.read().values().cloned().collect()
    }

    /// Number of known peers.
    pub fn len(&self) -> usize {
        self.known_peers.read().len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.known_peers.read().is_empty()
    }

    /// Get a specific peer by ID.
    pub fn get(&self, peer_id: &str) -> Option<ComputePeerInfo> {
        self.known_peers.read().get(peer_id).cloned()
    }

    /// Select the best peer for a given task type using score-based ranking.
    ///
    /// Scoring function:
    ///   `score = capability_match * (1.0 / latency_estimate_ms) * availability_ratio`
    ///
    /// - `capability_match`: How well the peer's resources fit the task
    ///   - "mining" / "gpu" -> weighted by gpu_tflops
    ///   - "inference" / "ai" -> weighted by ram_available_gb + gpu_tflops
    ///   - "zk" / "proof" -> weighted by available_cores
    ///   - default -> weighted by available_cores
    ///
    /// - `latency_estimate_ms`: Derived from bandwidth (higher bandwidth = lower latency).
    ///   We use `1000.0 / bandwidth_mbps` as a rough proxy since we don't have RTT data
    ///   until a tunnel is established.
    ///
    /// - `availability_ratio`: `available_cores / total_cores` — how idle the peer is.
    ///
    /// Returns `None` if no peers are known.
    pub fn get_best_peer_for_task(&self, task_type: &str) -> Option<ComputePeerInfo> {
        let peers = self.known_peers.read();
        if peers.is_empty() {
            return None;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let task_lower = task_type.to_lowercase();

        peers
            .values()
            // Filter out stale peers
            .filter(|p| now.saturating_sub(p.timestamp) <= PEER_TTL_SECS)
            // Filter out peers with no available capacity
            .filter(|p| p.available_cores > 0 || p.gpu_tflops > 0.0)
            .max_by(|a, b| {
                let score_a = compute_peer_score(a, &task_lower);
                let score_b = compute_peer_score(b, &task_lower);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }
}

/// Compute a score for a peer given a task type.
///
/// `score = capability_match * (1.0 / latency_estimate_ms) * availability_ratio`
fn compute_peer_score(peer: &ComputePeerInfo, task_type: &str) -> f64 {
    // Capability match — task-dependent weight
    let capability_match: f64 = match task_type {
        t if t.contains("mining") || t.contains("gpu") => {
            // GPU-heavy tasks: prefer peers with high TFLOPS
            peer.gpu_tflops.max(0.1) + (peer.available_cores as f64 * 0.1)
        }
        t if t.contains("inference") || t.contains("ai") => {
            // AI inference: needs RAM + GPU
            peer.ram_available_gb * 0.5 + peer.gpu_tflops * 2.0 + (peer.available_cores as f64 * 0.2)
        }
        t if t.contains("zk") || t.contains("proof") => {
            // ZK proofs: CPU-bound
            peer.available_cores as f64
        }
        _ => {
            // Generic tasks: balanced scoring
            peer.available_cores as f64 + peer.gpu_tflops * 0.5
        }
    };

    // Latency estimate: use inverse of bandwidth as a proxy for latency.
    // Higher bandwidth -> lower latency estimate -> higher score.
    // Clamp bandwidth to avoid division by zero.
    let bandwidth_clamped = peer.bandwidth_mbps.max(1.0);
    let latency_estimate_ms = 1000.0 / bandwidth_clamped;
    let latency_factor = 1.0 / latency_estimate_ms.max(0.01);

    // Availability ratio: how idle is this peer?
    let availability_ratio = if peer.total_cores > 0 {
        peer.available_cores as f64 / peer.total_cores as f64
    } else {
        0.0
    };

    capability_match * latency_factor * availability_ratio
}

// ═══════════════════════════════════════════════════════════════════
// Tunnel Manager — tracks all active tunnels to peers
// ═══════════════════════════════════════════════════════════════════

/// Tunnel manager — tracks all active tunnels to peers.
///
/// Supports two modes of opening a tunnel:
/// - **Legacy** (`open_tunnel`): Immediately transitions to `Active`.
/// - **Handshake** (`open_tunnel_with_handshake`): Starts in `Handshaking`,
///   sends a `TunnelHandshake`, waits for `HandshakeResponse`, then
///   transitions to `Active` on success.
///
/// The manager also exposes `auto_connect_to_best_peers()` which can be
/// called from a scheduler loop (e.g., every 30 seconds) to proactively
/// open tunnels to high-score discovered peers.
pub struct TunnelManager {
    tunnels: Arc<RwLock<HashMap<String, ComputeTunnel>>>,
    max_tunnels: usize,
    total_tasks_routed: Arc<AtomicU64>,
    /// Registry of known compute peers discovered via gossipsub
    peer_registry: Arc<PeerRegistry>,
    /// Our own peer ID — used as `initiator_peer_id` in handshakes.
    local_peer_id: String,
    /// Our advertised capabilities for handshake negotiation.
    local_capabilities: Vec<String>,
    /// Channel for sending handshake messages to the P2P transport layer.
    /// The transport layer reads from the corresponding receiver and
    /// publishes handshake messages to the peer.
    handshake_tx: mpsc::Sender<(String, Vec<u8>)>,
    /// Receiver is taken once by the transport layer.
    handshake_rx: RwLock<Option<mpsc::Receiver<(String, Vec<u8>)>>>,
}

impl TunnelManager {
    pub fn new(max_tunnels: usize) -> Self {
        let (handshake_tx, handshake_rx) = mpsc::channel(64);
        Self {
            tunnels: Arc::new(RwLock::new(HashMap::new())),
            max_tunnels,
            total_tasks_routed: Arc::new(AtomicU64::new(0)),
            peer_registry: Arc::new(PeerRegistry::new()),
            local_peer_id: String::new(),
            local_capabilities: vec![
                "mining".to_string(),
                "inference".to_string(),
                "bridge-verify".to_string(),
            ],
            handshake_tx,
            handshake_rx: RwLock::new(Some(handshake_rx)),
        }
    }

    /// Create a TunnelManager with a known local peer identity and
    /// capability list.
    pub fn with_identity(
        max_tunnels: usize,
        local_peer_id: String,
        capabilities: Vec<String>,
    ) -> Self {
        let (handshake_tx, handshake_rx) = mpsc::channel(64);
        Self {
            tunnels: Arc::new(RwLock::new(HashMap::new())),
            max_tunnels,
            total_tasks_routed: Arc::new(AtomicU64::new(0)),
            peer_registry: Arc::new(PeerRegistry::new()),
            local_peer_id,
            local_capabilities: capabilities,
            handshake_tx,
            handshake_rx: RwLock::new(Some(handshake_rx)),
        }
    }

    /// Take the handshake message receiver. Called once by the transport
    /// layer to drain outgoing handshake/response messages.
    /// Each item is `(target_peer_id, serialized_message_bytes)`.
    pub fn take_handshake_rx(&self) -> Option<mpsc::Receiver<(String, Vec<u8>)>> {
        self.handshake_rx.write().take()
    }

    /// Get a reference to the peer registry for direct access.
    pub fn peer_registry(&self) -> &Arc<PeerRegistry> {
        &self.peer_registry
    }

    /// Open a new tunnel to a peer (legacy — immediately Active).
    pub fn open_tunnel(&self, peer_id: &str, tunnel_type: TunnelType) -> bool {
        let mut tunnels = self.tunnels.write();

        if tunnels.len() >= self.max_tunnels {
            warn!(
                "🔗 [TUNNEL] Max tunnels reached ({}) — cannot open to {}",
                self.max_tunnels, peer_id
            );
            return false;
        }

        if tunnels.contains_key(peer_id) {
            debug!("🔗 [TUNNEL] Already connected to {}", peer_id);
            return true;
        }

        let tunnel = ComputeTunnel::new(peer_id.to_string(), tunnel_type);
        info!(
            "🔗 [TUNNEL] Opened {:?} tunnel to {} (encrypted={})",
            tunnel_type, peer_id, tunnel.encrypted
        );
        tunnels.insert(peer_id.to_string(), tunnel);
        true
    }

    /// Open a tunnel using the handshake protocol.
    ///
    /// 1. Creates the tunnel in `Handshaking` state.
    /// 2. Sends a `TunnelHandshake` to the peer via `handshake_tx`.
    /// 3. Returns the serialized handshake for the caller to deliver.
    ///
    /// The caller must later call `complete_handshake()` when a
    /// `HandshakeResponse` arrives from the peer.
    pub fn open_tunnel_with_handshake(
        &self,
        peer_id: &str,
        tunnel_type: TunnelType,
    ) -> Result<TunnelHandshake, String> {
        let mut tunnels = self.tunnels.write();

        if tunnels.len() >= self.max_tunnels {
            return Err(format!(
                "Max tunnels reached ({}) — cannot open to {}",
                self.max_tunnels, peer_id
            ));
        }

        if tunnels.contains_key(peer_id) {
            return Err(format!("Already have tunnel to {}", peer_id));
        }

        // Create tunnel in Handshaking state
        let tunnel = ComputeTunnel::with_state(
            peer_id.to_string(),
            tunnel_type,
            TunnelState::Handshaking,
        );
        tunnels.insert(peer_id.to_string(), tunnel);

        // Build the handshake message
        let handshake = TunnelHandshake::new(
            &self.local_peer_id,
            peer_id,
            self.local_capabilities.clone(),
        );

        info!(
            "🔗 [TUNNEL] Initiating handshake with {} (caps={:?}, proto=v{})",
            peer_id, handshake.capabilities, handshake.protocol_version,
        );

        // Queue the handshake for the transport layer
        let serialized = serde_json::to_vec(&handshake).unwrap_or_default();
        let _ = self.handshake_tx.try_send((peer_id.to_string(), serialized));

        Ok(handshake)
    }

    /// Process an incoming `HandshakeResponse` for a pending tunnel.
    ///
    /// If accepted, transitions the tunnel to `Active` and stores the
    /// negotiated capabilities. If rejected or the tunnel does not exist
    /// in `Handshaking` state, the tunnel is removed.
    pub fn complete_handshake(
        &self,
        peer_id: &str,
        response: &HandshakeResponse,
    ) -> Result<(), String> {
        let mut tunnels = self.tunnels.write();

        let tunnel = tunnels.get_mut(peer_id).ok_or_else(|| {
            format!("No pending tunnel for {}", peer_id)
        })?;

        {
            let current_st = tunnel.state.read().clone();
            if current_st != TunnelState::Handshaking {
                return Err(format!(
                    "Tunnel to {} is in {:?} state, not Handshaking",
                    peer_id, current_st
                ));
            }
        }

        if response.accepted {
            *tunnel.state.write() = TunnelState::Active;
            tunnel.negotiated_capabilities = response.capabilities.clone();
            info!(
                "🔗 [TUNNEL] Handshake accepted by {} (caps={:?})",
                peer_id, response.capabilities,
            );
            Ok(())
        } else {
            let reason = response.reason.as_deref().unwrap_or("unknown");
            warn!(
                "🔗 [TUNNEL] Handshake rejected by {}: {}",
                peer_id, reason,
            );
            tunnels.remove(peer_id);
            Err(format!("Handshake rejected: {}", reason))
        }
    }

    /// Handle an incoming `TunnelHandshake` from a remote peer who wants
    /// to open a tunnel to us. Returns a `HandshakeResponse` to send back.
    ///
    /// Acceptance criteria:
    /// - We have room for more tunnels.
    /// - Protocol version is compatible (currently must be 1).
    /// - The handshake is not stale (timestamp within 30 seconds).
    pub fn handle_incoming_handshake(
        &self,
        handshake: &TunnelHandshake,
    ) -> HandshakeResponse {
        // Check capacity
        let tunnels = self.tunnels.read();
        if tunnels.len() >= self.max_tunnels {
            return HandshakeResponse::reject("tunnel capacity full");
        }
        if tunnels.contains_key(&handshake.initiator_peer_id) {
            return HandshakeResponse::reject("tunnel already exists");
        }
        drop(tunnels);

        // Check protocol version
        if handshake.protocol_version != 1 {
            return HandshakeResponse::reject(&format!(
                "unsupported protocol version {}",
                handshake.protocol_version
            ));
        }

        // Check staleness (30 second window)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if now.saturating_sub(handshake.timestamp) > 30 {
            return HandshakeResponse::reject("handshake too old");
        }

        // Compute capability intersection
        let our_caps: std::collections::HashSet<&str> =
            self.local_capabilities.iter().map(|s| s.as_str()).collect();
        let shared_caps: Vec<String> = handshake
            .capabilities
            .iter()
            .filter(|c| our_caps.contains(c.as_str()))
            .cloned()
            .collect();

        // Accept — create the tunnel on our side too
        let mut tunnels = self.tunnels.write();
        let mut tunnel = ComputeTunnel::with_state(
            handshake.initiator_peer_id.clone(),
            TunnelType::NodeToNode,
            TunnelState::Active, // We go straight to Active as responder
        );
        tunnel.negotiated_capabilities = shared_caps.clone();
        tunnels.insert(handshake.initiator_peer_id.clone(), tunnel);

        info!(
            "🔗 [TUNNEL] Accepted incoming handshake from {} (shared_caps={:?})",
            handshake.initiator_peer_id, shared_caps,
        );

        HandshakeResponse::accept(shared_caps)
    }

    /// Time out handshakes that have been pending longer than
    /// `HANDSHAKE_TIMEOUT_SECS`. Called from the cleanup loop.
    pub fn timeout_stale_handshakes(&self) {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let timeout_ms = HANDSHAKE_TIMEOUT_SECS * 1000;

        let mut tunnels = self.tunnels.write();
        let stale: Vec<String> = tunnels
            .iter()
            .filter(|(_, t)| {
                *t.state.read() == TunnelState::Handshaking
                    && now_ms.saturating_sub(t.established_ms) > timeout_ms
            })
            .map(|(id, _)| id.clone())
            .collect();

        for peer_id in &stale {
            warn!(
                "🔗 [TUNNEL] Handshake timeout for {} (>{}s) — removing",
                peer_id, HANDSHAKE_TIMEOUT_SECS,
            );
            tunnels.remove(peer_id);
        }
    }

    /// Close tunnel to a peer
    pub fn close_tunnel(&self, peer_id: &str) {
        let mut tunnels = self.tunnels.write();
        if let Some(tunnel) = tunnels.remove(peer_id) {
            tunnel.close();
            info!(
                "🔗 [TUNNEL] Closed tunnel to {} (sent={}B, recv={}B, tasks={})",
                peer_id,
                tunnel.bytes_sent.load(Ordering::Relaxed),
                tunnel.bytes_received.load(Ordering::Relaxed),
                tunnel.tasks_routed.load(Ordering::Relaxed),
            );
        }
    }

    /// Gracefully drain a tunnel — transitions to `Draining` state so
    /// pending tasks finish but no new ones are accepted.
    pub fn drain_tunnel(&self, peer_id: &str) -> bool {
        let mut tunnels = self.tunnels.write();
        if let Some(tunnel) = tunnels.get_mut(peer_id) {
            tunnel.transition_to(TunnelState::Draining)
        } else {
            false
        }
    }

    /// Route a work item to the best available tunnel
    pub fn route_work(&self, item: &TunnelWorkItem) -> Option<String> {
        let tunnels = self.tunnels.read();

        // Find best tunnel: lowest latency active tunnel (must be in Active state)
        let best = tunnels.values()
            .filter(|t| t.is_active() && *t.state.read() == TunnelState::Active)
            .min_by_key(|t| t.latency_ms.load(Ordering::Relaxed));

        if let Some(tunnel) = best {
            tunnel.record_task();
            tunnel.record_send(item.payload_bytes as u64);
            self.total_tasks_routed.fetch_add(1, Ordering::Relaxed);
            debug!(
                "🔗 [TUNNEL] Routed {:?} task #{} ({} bytes) -> {}",
                item.layer, item.id, item.payload_bytes, tunnel.peer_id
            );
            Some(tunnel.peer_id.clone())
        } else {
            None
        }
    }

    /// Open a multiplexed stream on an existing tunnel.
    ///
    /// Returns the `stream_id` on success.
    pub fn open_stream_on_tunnel(
        &self,
        peer_id: &str,
        stream_type: StreamType,
    ) -> Result<u64, String> {
        let tunnels = self.tunnels.read();
        let tunnel = tunnels
            .get(peer_id)
            .ok_or_else(|| format!("No tunnel to {}", peer_id))?;
        tunnel.open_stream(stream_type)
    }

    /// Close a stream on an existing tunnel.
    pub fn close_stream_on_tunnel(&self, peer_id: &str, stream_id: u64) -> bool {
        let tunnels = self.tunnels.read();
        if let Some(tunnel) = tunnels.get(peer_id) {
            tunnel.close_stream(stream_id)
        } else {
            false
        }
    }

    /// Get all tunnel info snapshots for dashboard
    pub fn tunnel_infos(&self) -> Vec<TunnelInfo> {
        self.tunnels.read().values()
            .filter(|t| t.is_active())
            .map(|t| t.info())
            .collect()
    }

    /// Number of active tunnels
    pub fn active_count(&self) -> usize {
        self.tunnels.read().values()
            .filter(|t| t.is_active() && *t.state.read() == TunnelState::Active)
            .count()
    }

    /// Total tasks routed across all tunnels
    pub fn total_tasks_routed(&self) -> u64 {
        self.total_tasks_routed.load(Ordering::Relaxed)
    }

    /// Get the state of a tunnel to a specific peer.
    pub fn tunnel_state(&self, peer_id: &str) -> Option<TunnelState> {
        self.tunnels.read().get(peer_id).map(|t| t.state.read().clone())
    }

    /// Get the negotiated capabilities for a tunnel.
    pub fn tunnel_capabilities(&self, peer_id: &str) -> Option<Vec<String>> {
        self.tunnels
            .read()
            .get(peer_id)
            .map(|t| t.negotiated_capabilities.clone())
    }

    // ─── Auto-connect logic ──────────────────────────────────────

    /// Automatically open tunnels to the best discovered peers.
    ///
    /// Called from the scheduler loop (every ~30 seconds). Selects peers
    /// from the registry that:
    /// - Score above `min_score` for a generic task.
    /// - Do not already have a tunnel.
    ///
    /// Opens at most `MAX_AUTO_OPENS_PER_CALL` (2) new tunnels per
    /// invocation to avoid burst overhead. Returns the list of peers
    /// that handshakes were initiated with.
    pub fn auto_connect_to_best_peers(
        &self,
        min_score: f64,
        max_new_tunnels: usize,
    ) -> Vec<String> {
        let limit = max_new_tunnels.min(MAX_AUTO_OPENS_PER_CALL);

        let all_peers = self.peer_registry.all_peers();
        let tunnels = self.tunnels.read();

        // Score and filter peers
        let mut candidates: Vec<(f64, &ComputePeerInfo)> = all_peers
            .iter()
            .filter(|p| {
                // Skip peers we already have a tunnel to
                !tunnels.contains_key(&p.peer_id)
                    // Skip ourselves
                    && p.peer_id != self.local_peer_id
                    // Must have some capacity
                    && (p.available_cores > 0 || p.gpu_tflops > 0.0)
            })
            .map(|p| (compute_peer_score(p, "generic"), p))
            .filter(|(score, _)| *score >= min_score)
            .collect();

        // Sort descending by score — best peers first
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        drop(tunnels);

        let mut opened = Vec::new();
        for (score, peer) in candidates.into_iter().take(limit) {
            let peer_id = peer.peer_id.clone();
            match self.open_tunnel_with_handshake(&peer_id, TunnelType::NodeToNode) {
                Ok(_handshake) => {
                    info!(
                        "🔗 [AUTO-CONNECT] Initiated handshake with {} (score={:.2})",
                        peer_id, score,
                    );
                    opened.push(peer_id);
                }
                Err(e) => {
                    debug!(
                        "🔗 [AUTO-CONNECT] Skipped {}: {}",
                        peer_id, e,
                    );
                }
            }
        }

        if !opened.is_empty() {
            info!(
                "🔗 [AUTO-CONNECT] Initiated {} new tunnel handshakes",
                opened.len(),
            );
        }

        opened
    }

    /// Clean up dead tunnels, timed-out handshakes, and stale peers.
    pub fn cleanup_dead(&self) {
        // Time out stale handshakes first
        self.timeout_stale_handshakes();

        // Clean up dead/closed tunnels
        let mut tunnels = self.tunnels.write();
        let before = tunnels.len();
        tunnels.retain(|_, t| t.is_active());
        let removed = before - tunnels.len();
        if removed > 0 {
            info!("🔗 [TUNNEL] Cleaned up {} dead tunnels", removed);
        }
        drop(tunnels);

        // Clean up stale peers
        self.peer_registry.cleanup_stale();
    }
}

// NOTE: Crypto tunnel handshake (X25519 + HKDF) planned for Phase 2.
// Requires adding hkdf, sha2, and x25519-dalek to Cargo.toml.

/// State machine for the cryptographic tunnel handshake.
///
/// Progression: `Init -> KeyExchangeSent -> KeyExchangeReceived -> Established`
///
/// The handshake uses X25519 Diffie-Hellman for key exchange and HKDF-SHA256
/// to derive a 32-byte session key from the shared secret.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum HandshakeState {
    /// Initial state — no keys exchanged yet.
    Init,
    /// Our ephemeral public key has been sent to the peer.
    KeyExchangeSent,
    /// We received the peer's ephemeral public key.
    KeyExchangeReceived,
    /// Session key derived — tunnel is cryptographically established.
    Established,
}

impl std::fmt::Display for HandshakeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandshakeState::Init => write!(f, "Init"),
            HandshakeState::KeyExchangeSent => write!(f, "KeyExchangeSent"),
            HandshakeState::KeyExchangeReceived => write!(f, "KeyExchangeReceived"),
            HandshakeState::Established => write!(f, "Established"),
        }
    }
}

/// Initiator's key exchange message sent to the responder.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HandshakeInit {
    /// Initiator's peer ID (for correlation).
    pub initiator_peer_id: String,
    /// Ephemeral X25519 public key bytes (32 bytes).
    pub ephemeral_public_key: [u8; 32],
    /// Random nonce for replay protection.
    pub nonce: [u8; 32],
    /// Unix timestamp (seconds) of creation.
    pub timestamp: u64,
}

/// Responder's key exchange reply.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HandshakeKeyResponse {
    /// Responder's peer ID.
    pub responder_peer_id: String,
    /// Responder's ephemeral X25519 public key bytes (32 bytes).
    pub ephemeral_public_key: [u8; 32],
    /// Echo of the initiator's nonce (proves the responder saw our init).
    pub initiator_nonce: [u8; 32],
    /// Responder's own random nonce.
    pub nonce: [u8; 32],
    /// Unix timestamp (seconds).
    pub timestamp: u64,
}

/// A derived 32-byte session key used for symmetric encryption of tunnel traffic.
///
/// Implements `Zeroize` and `Drop` to ensure key material is securely erased
/// from memory when no longer needed (Issue #024 forward secrecy).
#[derive(Clone, Zeroize)]
#[zeroize(drop)]
pub struct SessionKey {
    /// The raw 32-byte key material derived from HKDF-SHA256.
    key: [u8; 32],
}

impl SessionKey {
    /// Access the raw key bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.key
    }

    /// Create a SessionKey from raw bytes. Used internally during rekey.
    fn from_bytes(key: [u8; 32]) -> Self {
        Self { key }
    }
}

impl std::fmt::Debug for SessionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never log key material
        write!(f, "SessionKey([REDACTED])")
    }
}

/// Manages the X25519 key exchange and HKDF session key derivation for a
/// single tunnel handshake.
///
/// Usage:
/// 1. Initiator calls `initiate_handshake(peer_id)` -> gets `HandshakeInit`.
/// 2. Send `HandshakeInit` to the peer.
/// 3. Peer creates their own `CryptoHandshake`, calls `respond_to_handshake(init)`.
/// 4. Initiator calls `complete_handshake(init, response)` -> gets `SessionKey`.
pub struct CryptoHandshake {
    /// Current state of the handshake state machine.
    state: HandshakeState,
    /// Our peer ID.
    local_peer_id: String,
    /// Our ephemeral X25519 secret key (consumed during key derivation).
    ephemeral_secret: Option<x25519_dalek::EphemeralSecret>,
    /// Our ephemeral public key (derived from the secret).
    ephemeral_public: Option<x25519_dalek::PublicKey>,
    /// The nonce we generated.
    our_nonce: [u8; 32],
    /// The derived session key (set once Established).
    session_key: Option<SessionKey>,
}

impl CryptoHandshake {
    /// Create a new handshake context for the given local peer.
    pub fn new(local_peer_id: &str) -> Self {
        let mut rng = rand::thread_rng();
        let secret = x25519_dalek::EphemeralSecret::random_from_rng(&mut rng);
        let public = x25519_dalek::PublicKey::from(&secret);

        let mut nonce = [0u8; 32];
        use rand::RngCore;
        rng.fill_bytes(&mut nonce);

        Self {
            state: HandshakeState::Init,
            local_peer_id: local_peer_id.to_string(),
            ephemeral_secret: Some(secret),
            ephemeral_public: Some(public),
            our_nonce: nonce,
            session_key: None,
        }
    }

    /// Create a CryptoHandshake from raw key material (for testing).
    #[cfg(test)]
    fn from_raw(local_peer_id: &str, secret_bytes: [u8; 32], nonce: [u8; 32]) -> Self {
        let secret = x25519_dalek::StaticSecret::from(secret_bytes);
        let public = x25519_dalek::PublicKey::from(&secret);
        // We need an EphemeralSecret for the real API, but for testing we
        // use StaticSecret and store the shared-secret derivation path
        // differently. Instead, we use the raw approach.
        Self {
            state: HandshakeState::Init,
            local_peer_id: local_peer_id.to_string(),
            ephemeral_secret: None,
            ephemeral_public: Some(public),
            our_nonce: nonce,
            // Store the static secret for test usage via a helper
            session_key: None,
        }
    }

    /// Current handshake state.
    pub fn state(&self) -> HandshakeState {
        self.state
    }

    /// Generate the initiation message to send to the peer.
    ///
    /// Transitions: `Init -> KeyExchangeSent`.
    pub fn initiate_handshake(&mut self, peer_id: &str) -> Result<HandshakeInit, String> {
        if self.state != HandshakeState::Init {
            return Err(format!(
                "Cannot initiate handshake from state {}",
                self.state
            ));
        }

        let public_key = self
            .ephemeral_public
            .ok_or("Ephemeral public key not available")?;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let init = HandshakeInit {
            initiator_peer_id: self.local_peer_id.clone(),
            ephemeral_public_key: public_key.to_bytes(),
            nonce: self.our_nonce,
            timestamp,
        };

        self.state = HandshakeState::KeyExchangeSent;
        debug!(
            "🔐 [HANDSHAKE] {} -> KeyExchangeSent (target={})",
            self.local_peer_id, peer_id
        );

        Ok(init)
    }

    /// Respond to an incoming handshake init from a remote peer.
    ///
    /// Performs the DH exchange and derives the session key immediately
    /// (the responder completes in one step).
    ///
    /// Transitions: `Init -> Established`.
    pub fn respond_to_handshake(
        &mut self,
        init: &HandshakeInit,
    ) -> Result<(HandshakeKeyResponse, SessionKey), String> {
        if self.state != HandshakeState::Init {
            return Err(format!(
                "Cannot respond to handshake from state {}",
                self.state
            ));
        }

        let our_public = self
            .ephemeral_public
            .ok_or("Ephemeral public key not available")?;
        let our_secret = self
            .ephemeral_secret
            .take()
            .ok_or("Ephemeral secret already consumed")?;

        // Perform DH with the initiator's public key
        let their_public = x25519_dalek::PublicKey::from(init.ephemeral_public_key);
        let shared_secret = our_secret.diffie_hellman(&their_public);

        // Derive session key via HKDF-SHA256
        let session_key = derive_session_key(
            shared_secret.as_bytes(),
            &init.nonce,
            &self.our_nonce,
        );

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let response = HandshakeKeyResponse {
            responder_peer_id: self.local_peer_id.clone(),
            ephemeral_public_key: our_public.to_bytes(),
            initiator_nonce: init.nonce,
            nonce: self.our_nonce,
            timestamp,
        };

        self.session_key = Some(session_key.clone());
        self.state = HandshakeState::Established;

        debug!(
            "🔐 [HANDSHAKE] {} responded to {} -> Established",
            self.local_peer_id, init.initiator_peer_id
        );

        Ok((response, session_key))
    }

    /// Complete the handshake on the initiator side after receiving the
    /// responder's key exchange response.
    ///
    /// Transitions: `KeyExchangeSent -> Established`.
    pub fn complete_handshake(
        &mut self,
        _init: &HandshakeInit,
        response: &HandshakeKeyResponse,
    ) -> Result<SessionKey, String> {
        if self.state != HandshakeState::KeyExchangeSent {
            return Err(format!(
                "Cannot complete handshake from state {}",
                self.state
            ));
        }

        // Verify the responder echoed our nonce
        if response.initiator_nonce != self.our_nonce {
            return Err("Nonce mismatch — possible replay attack".to_string());
        }

        let our_secret = self
            .ephemeral_secret
            .take()
            .ok_or("Ephemeral secret already consumed")?;

        // Perform DH with the responder's public key
        let their_public = x25519_dalek::PublicKey::from(response.ephemeral_public_key);
        let shared_secret = our_secret.diffie_hellman(&their_public);

        // Derive session key with the same salt construction as responder
        // (initiator_nonce, responder_nonce) — same order on both sides
        let session_key = derive_session_key(
            shared_secret.as_bytes(),
            &self.our_nonce,
            &response.nonce,
        );

        self.session_key = Some(session_key.clone());
        self.state = HandshakeState::Established;

        debug!(
            "🔐 [HANDSHAKE] {} completed with {} -> Established",
            self.local_peer_id, response.responder_peer_id
        );

        Ok(session_key)
    }

    /// Get the session key if the handshake has completed.
    pub fn session_key(&self) -> Option<&SessionKey> {
        self.session_key.as_ref()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tunnel Stats — per-tunnel statistics including rekey tracking
// (Issue #024 — Forward Secrecy)
// ═══════════════════════════════════════════════════════════════════

/// Statistics for a tunnel, including rekey tracking for forward secrecy.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TunnelStats {
    /// Total bytes sent through this tunnel.
    pub bytes_sent: u64,
    /// Total bytes received through this tunnel.
    pub bytes_received: u64,
    /// Total tasks routed through this tunnel.
    pub tasks_routed: u64,
    /// Current latency in milliseconds.
    pub latency_ms: u32,
    /// Number of successful rekey operations since tunnel establishment.
    pub rekey_count: u64,
    /// Unix timestamp (seconds) of the last successful rekey, or 0 if never rekeyed.
    pub last_rekey_time: u64,
    /// Bytes transferred since the last rekey (or since establishment if no rekey yet).
    pub bytes_since_rekey: u64,
}

// ═══════════════════════════════════════════════════════════════════
// Rekey State Machine (Issue #024 — Forward Secrecy)
// ═══════════════════════════════════════════════════════════════════

/// State of an in-progress rekey negotiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RekeyState {
    /// No rekey in progress — normal operation.
    Idle,
    /// We initiated a rekey and are waiting for the peer's RekeyAck.
    AwaitingAck,
    /// Rekey failed; will retry after `REKEY_RETRY_INTERVAL`.
    Failed,
}

/// Manages the periodic rekeying of a tunnel's session key to provide
/// forward secrecy.
///
/// The `RekeyManager` tracks:
/// - Bytes transferred since last rekey (to trigger on 10 GB threshold).
/// - Elapsed time since last rekey (to trigger on 1 hour threshold).
/// - Rekey state machine (Idle → AwaitingAck → Idle on success, or → Failed).
/// - The pending ephemeral secret used during an in-progress rekey.
///
/// ## Protocol
///
/// 1. **Initiator** calls `initiate_rekey()` → generates new X25519 keypair,
///    returns `TunnelPayload::Rekey` to send to the peer.
/// 2. **Responder** receives the `Rekey` payload, calls
///    `handle_rekey_request()` → generates their own keypair, derives
///    new session key, returns `TunnelPayload::RekeyAck`.
/// 3. **Initiator** receives `RekeyAck`, calls `complete_rekey()` → derives
///    the same new session key. Both sides atomically switch.
///
/// If either side fails, the old session key remains active and a retry is
/// scheduled after `REKEY_RETRY_INTERVAL`.
pub struct RekeyManager {
    /// Current rekey state.
    state: RekeyState,
    /// Bytes transferred since the last successful rekey.
    bytes_since_rekey: u64,
    /// When the last rekey completed (or tunnel was established).
    last_rekey_time: Instant,
    /// When a failed rekey occurred (for retry scheduling).
    last_failure_time: Option<Instant>,
    /// Counter of successful rekeys over the tunnel lifetime.
    rekey_count: u64,
    /// Monotonic sequence number for rekey correlation and replay prevention.
    next_rekey_seq: u64,
    /// Pending ephemeral secret during AwaitingAck state.
    /// Stored as StaticSecret so it can be cloned/held across the async gap.
    pending_secret: Option<x25519_dalek::StaticSecret>,
    /// Nonce we sent in the rekey initiation.
    pending_nonce: [u8; 32],
    /// The sequence number of the in-progress rekey.
    pending_seq: u64,
    /// Current session key — the active key used for encryption/decryption.
    /// Old keys are zeroized on replacement.
    current_session_key: Option<SessionKey>,
}

impl RekeyManager {
    /// Create a new `RekeyManager` starting from an initial session key.
    pub fn new(initial_key: SessionKey) -> Self {
        Self {
            state: RekeyState::Idle,
            bytes_since_rekey: 0,
            last_rekey_time: Instant::now(),
            last_failure_time: None,
            rekey_count: 0,
            next_rekey_seq: 1,
            pending_secret: None,
            pending_nonce: [0u8; 32],
            pending_seq: 0,
            current_session_key: Some(initial_key),
        }
    }

    /// Create a `RekeyManager` with no initial key (for testing).
    #[cfg(test)]
    pub fn new_empty() -> Self {
        Self {
            state: RekeyState::Idle,
            bytes_since_rekey: 0,
            last_rekey_time: Instant::now(),
            last_failure_time: None,
            rekey_count: 0,
            next_rekey_seq: 1,
            pending_secret: None,
            pending_nonce: [0u8; 32],
            pending_seq: 0,
            current_session_key: None,
        }
    }

    /// Current rekey state.
    pub fn state(&self) -> RekeyState {
        self.state
    }

    /// Number of successful rekeys over the tunnel lifetime.
    pub fn rekey_count(&self) -> u64 {
        self.rekey_count
    }

    /// Bytes transferred since the last rekey.
    pub fn bytes_since_rekey(&self) -> u64 {
        self.bytes_since_rekey
    }

    /// Get a reference to the current session key.
    pub fn current_key(&self) -> Option<&SessionKey> {
        self.current_session_key.as_ref()
    }

    /// Record bytes transferred through the tunnel. This is used to track
    /// the 10 GB threshold for auto-rekey.
    pub fn record_bytes(&mut self, bytes: u64) {
        self.bytes_since_rekey = self.bytes_since_rekey.saturating_add(bytes);
    }

    /// Check whether a rekey should be triggered based on time or byte thresholds.
    ///
    /// Returns `true` if:
    /// - We are in `Idle` state (no rekey in progress), AND
    /// - Either 1 hour has elapsed OR 10 GB has been transferred, AND
    /// - If we previously failed, at least 5 minutes have passed since the failure.
    pub fn should_rekey(&self) -> bool {
        if self.state != RekeyState::Idle {
            return false;
        }

        // Respect retry backoff on failure
        if let Some(failure_time) = self.last_failure_time {
            if failure_time.elapsed() < REKEY_RETRY_INTERVAL {
                return false;
            }
        }

        let time_exceeded = self.last_rekey_time.elapsed() >= REKEY_INTERVAL;
        let bytes_exceeded = self.bytes_since_rekey >= REKEY_BYTE_THRESHOLD;

        time_exceeded || bytes_exceeded
    }

    /// Initiate a rekey by generating a new ephemeral X25519 keypair.
    ///
    /// Returns the `TunnelPayload::Rekey` message to send to the peer.
    /// Transitions state from `Idle` to `AwaitingAck`.
    ///
    /// Returns `Err` if a rekey is already in progress.
    pub fn initiate_rekey(&mut self) -> Result<TunnelPayload, String> {
        if self.state != RekeyState::Idle {
            return Err(format!(
                "Cannot initiate rekey from state {:?}",
                self.state
            ));
        }

        let mut rng = rand::thread_rng();

        // Generate new ephemeral keypair. We use StaticSecret here (not
        // EphemeralSecret) because we need to hold it across the async gap
        // until the peer responds with RekeyAck.
        let secret_bytes = {
            let mut bytes = [0u8; 32];
            use rand::RngCore;
            rng.fill_bytes(&mut bytes);
            bytes
        };
        let secret = x25519_dalek::StaticSecret::from(secret_bytes);
        let public = x25519_dalek::PublicKey::from(&secret);

        let mut nonce = [0u8; 32];
        use rand::RngCore;
        rng.fill_bytes(&mut nonce);

        let seq = self.next_rekey_seq;
        self.next_rekey_seq += 1;

        self.pending_secret = Some(secret);
        self.pending_nonce = nonce;
        self.pending_seq = seq;
        self.state = RekeyState::AwaitingAck;

        info!(
            "🔑 [REKEY] Initiated rekey seq={} (bytes_since={}, elapsed={:.0}s)",
            seq,
            self.bytes_since_rekey,
            self.last_rekey_time.elapsed().as_secs_f64(),
        );

        Ok(TunnelPayload::Rekey {
            ephemeral_pubkey: public.to_bytes(),
            nonce,
            rekey_seq: seq,
        })
    }

    /// Handle an incoming rekey request from the remote peer.
    ///
    /// Generates a new ephemeral keypair, derives the new session key, and
    /// returns a `TunnelPayload::RekeyAck` to send back.
    ///
    /// The new session key is installed immediately on the responder side.
    /// The old key is zeroized.
    pub fn handle_rekey_request(
        &mut self,
        their_pubkey: &[u8; 32],
        their_nonce: &[u8; 32],
        rekey_seq: u64,
    ) -> Result<TunnelPayload, String> {
        let mut rng = rand::thread_rng();

        let secret_bytes = {
            let mut bytes = [0u8; 32];
            use rand::RngCore;
            rng.fill_bytes(&mut bytes);
            bytes
        };
        let secret = x25519_dalek::StaticSecret::from(secret_bytes);
        let public = x25519_dalek::PublicKey::from(&secret);

        let mut our_nonce = [0u8; 32];
        use rand::RngCore;
        rng.fill_bytes(&mut our_nonce);

        // Perform DH with the initiator's new public key
        let their_public = x25519_dalek::PublicKey::from(*their_pubkey);
        let shared_secret = secret.diffie_hellman(&their_public);

        // Derive new session key with the rekey-specific info string
        let new_key = derive_rekey_session_key(
            shared_secret.as_bytes(),
            their_nonce,
            &our_nonce,
            rekey_seq,
        );

        // Atomically replace the session key; old one is zeroized on drop
        self.current_session_key = Some(new_key);
        self.bytes_since_rekey = 0;
        self.last_rekey_time = Instant::now();
        self.rekey_count += 1;
        self.last_failure_time = None;

        info!(
            "🔑 [REKEY] Responded to rekey seq={} (total_rekeys={})",
            rekey_seq, self.rekey_count,
        );

        Ok(TunnelPayload::RekeyAck {
            ephemeral_pubkey: public.to_bytes(),
            nonce: our_nonce,
            rekey_seq,
        })
    }

    /// Complete a rekey on the initiator side after receiving the peer's
    /// `RekeyAck`.
    ///
    /// Derives the new shared session key and installs it. The old key is
    /// zeroized.
    ///
    /// Transitions state from `AwaitingAck` to `Idle`.
    pub fn complete_rekey(
        &mut self,
        their_pubkey: &[u8; 32],
        their_nonce: &[u8; 32],
        rekey_seq: u64,
    ) -> Result<(), String> {
        if self.state != RekeyState::AwaitingAck {
            return Err(format!(
                "Cannot complete rekey from state {:?}",
                self.state
            ));
        }

        if rekey_seq != self.pending_seq {
            return Err(format!(
                "Rekey seq mismatch: expected {}, got {}",
                self.pending_seq, rekey_seq
            ));
        }

        let secret = self.pending_secret.take().ok_or(
            "Pending ephemeral secret missing — rekey state corrupted"
        )?;

        let their_public = x25519_dalek::PublicKey::from(*their_pubkey);
        let shared_secret = secret.diffie_hellman(&their_public);

        // Derive new session key with the same construction as the responder
        let new_key = derive_rekey_session_key(
            shared_secret.as_bytes(),
            &self.pending_nonce,
            their_nonce,
            rekey_seq,
        );

        // Atomically replace the session key; old one is zeroized on drop
        self.current_session_key = Some(new_key);
        self.bytes_since_rekey = 0;
        self.last_rekey_time = Instant::now();
        self.rekey_count += 1;
        self.state = RekeyState::Idle;
        self.last_failure_time = None;

        info!(
            "🔑 [REKEY] Completed rekey seq={} (total_rekeys={})",
            rekey_seq, self.rekey_count,
        );

        Ok(())
    }

    /// Mark the current rekey as failed. The old session key is preserved
    /// and a retry will be attempted after `REKEY_RETRY_INTERVAL`.
    pub fn fail_rekey(&mut self, reason: &str) {
        warn!(
            "🔑 [REKEY] Rekey failed (seq={}): {} — keeping old key, retry in {}s",
            self.pending_seq,
            reason,
            REKEY_RETRY_INTERVAL.as_secs(),
        );

        // Clean up pending state
        self.pending_secret = None;
        self.pending_nonce.zeroize();
        self.state = RekeyState::Failed;
        self.last_failure_time = Some(Instant::now());
    }

    /// Transition from Failed back to Idle so the next `should_rekey()` check
    /// can trigger a retry. This is called automatically when the retry
    /// interval has elapsed.
    pub fn reset_failure(&mut self) {
        if self.state == RekeyState::Failed {
            self.state = RekeyState::Idle;
            debug!("🔑 [REKEY] Reset failure state — eligible for retry");
        }
    }

    /// Produce a `TunnelStats` snapshot including rekey information.
    pub fn stats(&self, bytes_sent: u64, bytes_received: u64, tasks_routed: u64, latency_ms: u32) -> TunnelStats {
        let last_rekey_time = if self.rekey_count > 0 {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                .saturating_sub(self.last_rekey_time.elapsed().as_secs())
        } else {
            0
        };

        TunnelStats {
            bytes_sent,
            bytes_received,
            tasks_routed,
            latency_ms,
            rekey_count: self.rekey_count,
            last_rekey_time,
            bytes_since_rekey: self.bytes_since_rekey,
        }
    }
}

/// Derive a session key specifically for rekey operations.
///
/// Uses a different HKDF info string (`"qnk-compute-tunnel-rekey-v1"`) than
/// the initial handshake to ensure domain separation. The rekey sequence
/// number is mixed into the salt for additional replay protection.
fn derive_rekey_session_key(
    shared_secret: &[u8],
    initiator_nonce: &[u8; 32],
    responder_nonce: &[u8; 32],
    rekey_seq: u64,
) -> SessionKey {
    use hkdf::Hkdf;
    use sha2::Sha256;

    // Salt = initiator_nonce || responder_nonce || rekey_seq (BE)
    let mut salt = [0u8; 72];
    salt[..32].copy_from_slice(initiator_nonce);
    salt[32..64].copy_from_slice(responder_nonce);
    salt[64..72].copy_from_slice(&rekey_seq.to_be_bytes());

    let hk = Hkdf::<Sha256>::new(Some(&salt), shared_secret);
    let mut okm = [0u8; 32];
    hk.expand(b"qnk-compute-tunnel-rekey-v1", &mut okm)
        .expect("HKDF expand should not fail for 32-byte output");

    SessionKey { key: okm }
}

/// Derive a 32-byte session key from a shared secret using HKDF-SHA256.
///
/// The salt is constructed by concatenating `initiator_nonce || responder_nonce`
/// so that both sides produce the same key regardless of who calls this function.
fn derive_session_key(
    shared_secret: &[u8],
    initiator_nonce: &[u8; 32],
    responder_nonce: &[u8; 32],
) -> SessionKey {
    use hkdf::Hkdf;
    use sha2::Sha256;

    // Salt = initiator_nonce || responder_nonce (deterministic order)
    let mut salt = [0u8; 64];
    salt[..32].copy_from_slice(initiator_nonce);
    salt[32..].copy_from_slice(responder_nonce);

    let hk = Hkdf::<Sha256>::new(Some(&salt), shared_secret);
    let mut okm = [0u8; 32];
    hk.expand(b"qnk-compute-tunnel-v1", &mut okm)
        .expect("HKDF expand should not fail for 32-byte output");

    SessionKey { key: okm }
}

// ═══════════════════════════════════════════════════════════════════
// Multiplexed Stream Framing Layer
// (Issue #002 — Criterion 5: Multiplexed Stream Types)
// ═══════════════════════════════════════════════════════════════════

/// Stream type for the framing layer — classifies the logical channel.
///
/// This is separate from `StreamType` (which is used by the higher-level
/// `TunnelStream`) to cleanly separate the framing concern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FrameStreamType {
    Mining,
    Inference,
    Proof,
    Control,
}

impl FrameStreamType {
    /// Encode the stream type as a single byte for frame headers.
    pub fn to_byte(&self) -> u8 {
        match self {
            FrameStreamType::Mining => 0,
            FrameStreamType::Inference => 1,
            FrameStreamType::Proof => 2,
            FrameStreamType::Control => 3,
        }
    }

    /// Decode a stream type from a byte.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(FrameStreamType::Mining),
            1 => Some(FrameStreamType::Inference),
            2 => Some(FrameStreamType::Proof),
            3 => Some(FrameStreamType::Control),
            _ => None,
        }
    }
}

impl std::fmt::Display for FrameStreamType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameStreamType::Mining => write!(f, "mining"),
            FrameStreamType::Inference => write!(f, "inference"),
            FrameStreamType::Proof => write!(f, "proof"),
            FrameStreamType::Control => write!(f, "control"),
        }
    }
}

/// Maximum frame payload size (1 MB). Payloads larger than this must be
/// split across multiple frames.
const MAX_FRAME_PAYLOAD: usize = 1_048_576;

/// Size of the frame header: `stream_id (4) + length (4)` = 8 bytes.
pub const FRAME_HEADER_SIZE: usize = 8;

/// A handle to a logical stream within the multiplexed tunnel.
///
/// The caller uses this handle to send and receive data on a specific
/// stream identified by `stream_id`.
#[derive(Debug, Clone)]
pub struct StreamHandle {
    pub stream_id: u32,
    pub stream_type: FrameStreamType,
}

/// Multiplexed framing layer over a single tunnel connection.
///
/// Provides logical channels (streams) over one physical connection.
/// Each frame has the wire format:
///
/// ```text
/// ┌──────────────┬──────────────┬─────────────────────┐
/// │ stream_id    │ length       │ payload              │
/// │ (4 bytes BE) │ (4 bytes BE) │ (length bytes)       │
/// └──────────────┴──────────────┴─────────────────────┘
/// ```
///
/// This is just the framing layer. Actual yamux integration happens when
/// wired to libp2p.
pub struct FramedTunnelStream {
    /// Open streams keyed by stream_id.
    streams: HashMap<u32, StreamHandle>,
    /// Outbound frame buffer — frames waiting to be flushed to transport.
    outbound: Vec<Vec<u8>>,
    /// Inbound per-stream buffers — frames received and dispatched.
    inbound: HashMap<u32, Vec<Vec<u8>>>,
    /// Next stream ID to assign.
    next_stream_id: u32,
}

impl FramedTunnelStream {
    /// Create a new framing layer.
    pub fn new() -> Self {
        Self {
            streams: HashMap::new(),
            outbound: Vec::new(),
            inbound: HashMap::new(),
            next_stream_id: 1,
        }
    }

    /// Open a new logical stream of the given type.
    ///
    /// Returns a `StreamHandle` that can be used with `send_on_stream`
    /// and `recv_on_stream`.
    pub fn open_stream(&mut self, stream_type: FrameStreamType) -> StreamHandle {
        let stream_id = self.next_stream_id;
        self.next_stream_id += 1;

        let handle = StreamHandle {
            stream_id,
            stream_type,
        };

        self.streams.insert(stream_id, handle.clone());
        self.inbound.insert(stream_id, Vec::new());

        debug!(
            "🔗 [FRAME] Opened {} stream #{}",
            stream_type, stream_id
        );

        handle
    }

    /// Close a stream by handle. Removes it from the internal tables.
    pub fn close_stream(&mut self, handle: &StreamHandle) {
        self.streams.remove(&handle.stream_id);
        self.inbound.remove(&handle.stream_id);
        debug!(
            "🔗 [FRAME] Closed {} stream #{}",
            handle.stream_type, handle.stream_id
        );
    }

    /// Encode and queue a payload for sending on the given stream.
    ///
    /// The payload is framed as `[stream_id: u32 BE][length: u32 BE][payload]`.
    /// Returns an error if the payload exceeds `MAX_FRAME_PAYLOAD` or the
    /// stream does not exist.
    pub fn send_on_stream(&mut self, handle: &StreamHandle, payload: &[u8]) -> Result<(), String> {
        if !self.streams.contains_key(&handle.stream_id) {
            return Err(format!("Stream #{} not found", handle.stream_id));
        }
        if payload.len() > MAX_FRAME_PAYLOAD {
            return Err(format!(
                "Payload too large: {} bytes (max {})",
                payload.len(),
                MAX_FRAME_PAYLOAD
            ));
        }

        let frame = encode_frame(handle.stream_id, payload);
        self.outbound.push(frame);
        Ok(())
    }

    /// Retrieve the next received payload for the given stream.
    ///
    /// Returns `Ok(payload)` if data is available, or an error if the stream
    /// does not exist or no data is queued.
    pub fn recv_on_stream(&mut self, handle: &StreamHandle) -> Result<Vec<u8>, String> {
        let queue = self
            .inbound
            .get_mut(&handle.stream_id)
            .ok_or_else(|| format!("Stream #{} not found", handle.stream_id))?;

        if queue.is_empty() {
            return Err("No data available on stream".to_string());
        }

        Ok(queue.remove(0))
    }

    /// Feed raw bytes from the transport into the framing layer for
    /// demultiplexing.
    ///
    /// Parses frames from `data` and dispatches payloads to the
    /// appropriate per-stream inbound queue. Returns the number of
    /// frames successfully parsed.
    pub fn feed_incoming(&mut self, data: &[u8]) -> usize {
        let mut offset = 0;
        let mut count = 0;

        while offset + FRAME_HEADER_SIZE <= data.len() {
            match decode_frame(&data[offset..]) {
                Some((stream_id, payload, consumed)) => {
                    if let Some(queue) = self.inbound.get_mut(&stream_id) {
                        queue.push(payload);
                    } else {
                        debug!(
                            "🔗 [FRAME] Received frame for unknown stream #{} — dropping",
                            stream_id
                        );
                    }
                    offset += consumed;
                    count += 1;
                }
                None => break,
            }
        }

        count
    }

    /// Drain all queued outbound frames. Returns the concatenated wire bytes.
    pub fn flush_outbound(&mut self) -> Vec<u8> {
        let total_size: usize = self.outbound.iter().map(|f| f.len()).sum();
        let mut out = Vec::with_capacity(total_size);
        for frame in self.outbound.drain(..) {
            out.extend_from_slice(&frame);
        }
        out
    }

    /// Number of currently open streams.
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Number of frames pending in the outbound buffer.
    pub fn outbound_pending(&self) -> usize {
        self.outbound.len()
    }
}

/// Encode a single frame: `[stream_id: u32 BE][length: u32 BE][payload]`.
pub fn encode_frame(stream_id: u32, payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(FRAME_HEADER_SIZE + payload.len());
    frame.extend_from_slice(&stream_id.to_be_bytes());
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(payload);
    frame
}

/// Decode a single frame from the beginning of `data`.
///
/// Returns `(stream_id, payload, total_bytes_consumed)` or `None` if the
/// buffer is too short for a complete frame.
pub fn decode_frame(data: &[u8]) -> Option<(u32, Vec<u8>, usize)> {
    if data.len() < FRAME_HEADER_SIZE {
        return None;
    }

    let stream_id = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    let length = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;

    if length > MAX_FRAME_PAYLOAD {
        warn!(
            "🔗 [FRAME] Frame length {} exceeds max {} — dropping",
            length, MAX_FRAME_PAYLOAD
        );
        return None;
    }

    let total = FRAME_HEADER_SIZE + length;
    if data.len() < total {
        return None; // Incomplete frame — need more data
    }

    let payload = data[FRAME_HEADER_SIZE..total].to_vec();
    Some((stream_id, payload, total))
}

// ═══════════════════════════════════════════════════════════════════
// Result Verification — 2-of-3 redundant compute
// (Issue #002 — Criterion 6: Result Verification)
// ═══════════════════════════════════════════════════════════════════

/// Timeout for individual peer responses in seconds.
const VERIFICATION_TIMEOUT_SECS: u64 = 10;

/// Minimum number of agreeing peers to accept a result.
const MIN_AGREEMENT: usize = 2;

/// Information about a compute peer used for task dispatch.
#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub peer_id: String,
}

/// The outcome of a verified computation — either accepted (2+ peers
/// agree) or rejected (all results differ / insufficient responses).
#[derive(Debug, Clone)]
pub struct VerifiedResult {
    /// The accepted result bytes (empty if rejected).
    pub result: Vec<u8>,
    /// Agreement ratio: fraction of responding peers that produced the
    /// winning result (e.g. 1.0 = unanimous, 0.67 = 2 of 3).
    pub agreement: f64,
    /// Peer IDs that participated (responded within timeout).
    pub participating_peers: Vec<String>,
    /// Whether the result was accepted (>= 2 agreeing).
    pub accepted: bool,
}

/// Dispatches a task to multiple peers and accepts the result only if a
/// quorum agrees.
///
/// The verifier is decoupled from the actual network transport. Callers
/// provide a `dispatch_fn` closure that sends the task to a peer and
/// returns the result bytes. The verifier handles timeouts, comparison,
/// and quorum logic.
pub struct ResultVerifier;

impl ResultVerifier {
    /// Submit a task for verified execution across multiple peers.
    ///
    /// The `dispatch_fn` is called once per peer. It receives `(peer_id, payload)`
    /// and should return the result bytes. The function is wrapped in a timeout
    /// of `VERIFICATION_TIMEOUT_SECS`.
    ///
    /// Returns a `VerifiedResult` with the quorum outcome.
    pub async fn submit_verified_task<F, Fut>(
        task: &TunnelPayload,
        peers: &[PeerInfo],
        dispatch_fn: F,
    ) -> VerifiedResult
    where
        F: Fn(String, TunnelPayload) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Vec<u8>, String>> + Send + 'static,
    {
        if peers.is_empty() {
            warn!("🔍 [VERIFIER] No peers provided for verified task");
            return VerifiedResult {
                result: Vec::new(),
                agreement: 0.0,
                participating_peers: Vec::new(),
                accepted: false,
            };
        }

        let dispatch_fn = Arc::new(dispatch_fn);
        let mut handles = Vec::new();

        // Dispatch to all peers concurrently
        for peer in peers {
            let peer_id = peer.peer_id.clone();
            let payload = task.clone();
            let dispatch = dispatch_fn.clone();

            let handle = tokio::spawn(async move {
                let result = tokio::time::timeout(
                    Duration::from_secs(VERIFICATION_TIMEOUT_SECS),
                    dispatch(peer_id.clone(), payload),
                )
                .await;

                match result {
                    Ok(Ok(data)) => {
                        debug!(
                            "🔍 [VERIFIER] Received {}B result from {}",
                            data.len(),
                            peer_id
                        );
                        Some((peer_id, data))
                    }
                    Ok(Err(e)) => {
                        warn!("🔍 [VERIFIER] Peer {} returned error: {}", peer_id, e);
                        None
                    }
                    Err(_) => {
                        warn!(
                            "🔍 [VERIFIER] Peer {} timed out after {}s",
                            peer_id, VERIFICATION_TIMEOUT_SECS
                        );
                        None
                    }
                }
            });

            handles.push(handle);
        }

        // Collect results
        let mut results: Vec<(String, Vec<u8>)> = Vec::new();
        for handle in handles {
            if let Ok(Some((peer_id, data))) = handle.await {
                results.push((peer_id, data));
            }
        }

        // Determine quorum
        Self::determine_quorum(results)
    }

    /// Synchronous verification from pre-collected results.
    ///
    /// Useful for testing and for cases where results have already been
    /// gathered through another mechanism.
    pub fn verify_results(results: Vec<(String, Vec<u8>)>) -> VerifiedResult {
        Self::determine_quorum(results)
    }

    /// Core quorum logic: find the most common result and check if it
    /// meets the minimum agreement threshold.
    fn determine_quorum(results: Vec<(String, Vec<u8>)>) -> VerifiedResult {
        let participating_peers: Vec<String> =
            results.iter().map(|(peer, _)| peer.clone()).collect();
        let total_responses = results.len();

        if total_responses == 0 {
            return VerifiedResult {
                result: Vec::new(),
                agreement: 0.0,
                participating_peers,
                accepted: false,
            };
        }

        // Group results by content — find the most common one.
        // We use a simple Vec-based approach since we expect <= 3 distinct results.
        let mut groups: Vec<(Vec<u8>, Vec<String>)> = Vec::new();
        for (peer_id, data) in &results {
            if let Some(group) = groups.iter_mut().find(|(d, _)| d == data) {
                group.1.push(peer_id.clone());
            } else {
                groups.push((data.clone(), vec![peer_id.clone()]));
            }
        }

        // Sort by group size descending
        groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

        let (best_result, best_peers) = &groups[0];
        let agreement = best_peers.len() as f64 / total_responses as f64;
        let accepted = best_peers.len() >= MIN_AGREEMENT;

        if accepted {
            info!(
                "🔍 [VERIFIER] Result accepted: {}/{} peers agree ({:.0}%)",
                best_peers.len(),
                total_responses,
                agreement * 100.0
            );
        } else {
            warn!(
                "🔍 [VERIFIER] Result REJECTED: only {}/{} peers agree (need {})",
                best_peers.len(),
                total_responses,
                MIN_AGREEMENT
            );
        }

        VerifiedResult {
            result: if accepted {
                best_result.clone()
            } else {
                Vec::new()
            },
            agreement,
            participating_peers,
            accepted,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tunnel_creation() {
        let tunnel = ComputeTunnel::new("peer123".to_string(), TunnelType::MinerToNode);
        assert!(tunnel.is_active());
        assert_eq!(tunnel.peer_id, "peer123");
        assert!(tunnel.encrypted);
    }

    #[test]
    fn test_tunnel_stats() {
        let tunnel = ComputeTunnel::new("peer456".to_string(), TunnelType::NodeToNode);
        tunnel.record_send(1000);
        tunnel.record_send(2000);
        tunnel.record_receive(500);
        tunnel.record_task();
        tunnel.record_task();
        tunnel.update_latency(42);

        let info = tunnel.info();
        assert_eq!(info.bytes_sent, 3000);
        assert_eq!(info.bytes_received, 500);
        assert_eq!(info.tasks_routed, 2);
        assert_eq!(info.latency_ms, 42);
    }

    #[test]
    fn test_tunnel_manager() {
        let mgr = TunnelManager::new(10);
        assert!(mgr.open_tunnel("peer1", TunnelType::MinerToNode));
        assert!(mgr.open_tunnel("peer2", TunnelType::NodeToNode));
        assert_eq!(mgr.active_count(), 2);

        // Duplicate returns true but doesn't add
        assert!(mgr.open_tunnel("peer1", TunnelType::MinerToNode));
        assert_eq!(mgr.active_count(), 2);

        mgr.close_tunnel("peer1");
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_tunnel_max_limit() {
        let mgr = TunnelManager::new(2);
        assert!(mgr.open_tunnel("peer1", TunnelType::MinerToNode));
        assert!(mgr.open_tunnel("peer2", TunnelType::NodeToNode));
        assert!(!mgr.open_tunnel("peer3", TunnelType::MinerToMiner)); // Exceeds max
        assert_eq!(mgr.active_count(), 2);
    }

    #[test]
    fn test_route_work() {
        let mgr = TunnelManager::new(10);
        mgr.open_tunnel("peer1", TunnelType::NodeToNode);

        let item = TunnelWorkItem {
            id: 1,
            layer: ComputeLayer::Mining,
            payload_bytes: 256,
            priority: 0,
            sender_peer: "local".to_string(),
        };

        let routed_to = mgr.route_work(&item);
        assert_eq!(routed_to, Some("peer1".to_string()));
        assert_eq!(mgr.total_tasks_routed(), 1);
    }

    // ═══════════════════════════════════════════════════════════════
    // Peer announcement tests
    // ═══════════════════════════════════════════════════════════════

    fn make_test_snapshot() -> ResourceSnapshot {
        ResourceSnapshot {
            cpu_per_core: vec![10.0, 20.0, 30.0, 90.0], // 3 cores < 80%, 1 >= 80%
            cpu_total: 37.5,
            gpu_utilization: 50.0,
            gpu_memory_used: 4 * 1024 * 1024 * 1024,     // 4 GB used
            gpu_memory_total: 8 * 1024 * 1024 * 1024,     // 8 GB total
            gpu_temperature: 72.0,
            gpu_name: "NVIDIA GeForce RTX 3080".to_string(),
            ram_used: 8 * 1024 * 1024 * 1024,              // 8 GB used
            ram_total: 32 * 1024 * 1024 * 1024,            // 32 GB total
            net_tx_bps: 10_000_000,
            net_rx_bps: 10_000_000,
            net_capacity_bps: 125_000_000, // 1 Gbps
            disk_io_bps: 100_000_000,
            timestamp_ms: 1710000000000,
        }
    }

    #[test]
    fn test_create_peer_announcement() {
        let snap = make_test_snapshot();
        let info = create_peer_announcement(&snap, "full", "12D3KooWTest123");

        assert_eq!(info.peer_id, "12D3KooWTest123");
        assert_eq!(info.total_cores, 4);
        assert_eq!(info.available_cores, 3); // 3 cores under 80%
        assert!(info.gpu_tflops > 0.0);
        assert!(info.ram_total_gb > 31.0 && info.ram_total_gb < 33.0);
        assert!(info.ram_available_gb > 23.0 && info.ram_available_gb < 25.0);
        assert_eq!(info.compute_mode, "full");
        assert!(info.bandwidth_mbps > 900.0); // ~1000 Mbps
        assert!(info.timestamp > 0);
        assert!(!info.version.is_empty());
    }

    #[test]
    fn test_create_announcement_no_gpu() {
        let mut snap = make_test_snapshot();
        snap.gpu_memory_total = 0;
        snap.gpu_utilization = 0.0;

        let info = create_peer_announcement(&snap, "eco", "peer-no-gpu");
        assert_eq!(info.gpu_tflops, 0.0);
        assert_eq!(info.compute_mode, "eco");
    }

    #[test]
    fn test_parse_peer_announcement_valid() {
        let snap = make_test_snapshot();
        let info = create_peer_announcement(&snap, "full", "12D3KooWTest");

        let json = serde_json::to_vec(&info).unwrap();
        let parsed = parse_peer_announcement(&json);
        assert!(parsed.is_some());

        let parsed = parsed.unwrap();
        assert_eq!(parsed.peer_id, "12D3KooWTest");
        assert_eq!(parsed.total_cores, 4);
        assert_eq!(parsed.compute_mode, "full");
    }

    #[test]
    fn test_parse_peer_announcement_invalid() {
        assert!(parse_peer_announcement(b"not json").is_none());
        assert!(parse_peer_announcement(b"{}").is_none()); // Missing required fields
        assert!(parse_peer_announcement(b"").is_none());
    }

    // ═══════════════════════════════════════════════════════════════
    // Peer registry tests
    // ═══════════════════════════════════════════════════════════════

    fn make_test_peer(peer_id: &str, cores: u32, gpu: f64, ram_gb: f64, bw: f64) -> ComputePeerInfo {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        ComputePeerInfo {
            peer_id: peer_id.to_string(),
            available_cores: cores,
            total_cores: cores * 2,
            gpu_tflops: gpu,
            ram_available_gb: ram_gb,
            ram_total_gb: ram_gb * 2.0,
            bandwidth_mbps: bw,
            compute_mode: "full".to_string(),
            active_layers: vec!["Mining".to_string()],
            trainer_active: false,
            version: "test".to_string(),
            timestamp: now,
        }
    }

    #[test]
    fn test_peer_registry_upsert_and_get() {
        let registry = PeerRegistry::new();
        assert!(registry.is_empty());

        let peer = make_test_peer("peer-A", 8, 10.0, 16.0, 1000.0);
        registry.upsert(peer);

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());

        let fetched = registry.get("peer-A");
        assert!(fetched.is_some());
        assert_eq!(fetched.unwrap().available_cores, 8);

        // Not found
        assert!(registry.get("peer-B").is_none());
    }

    #[test]
    fn test_peer_registry_update_existing() {
        let registry = PeerRegistry::new();

        let peer_v1 = make_test_peer("peer-A", 4, 5.0, 8.0, 500.0);
        registry.upsert(peer_v1);
        assert_eq!(registry.get("peer-A").unwrap().available_cores, 4);

        // Update with new data
        let peer_v2 = make_test_peer("peer-A", 8, 10.0, 16.0, 1000.0);
        registry.upsert(peer_v2);
        assert_eq!(registry.len(), 1); // Same peer, no duplicate
        assert_eq!(registry.get("peer-A").unwrap().available_cores, 8);
    }

    #[test]
    fn test_peer_registry_stale_rejected() {
        let registry = PeerRegistry::new();

        let mut stale_peer = make_test_peer("stale", 4, 0.0, 8.0, 100.0);
        stale_peer.timestamp = 1000; // Very old timestamp
        registry.upsert(stale_peer);

        // Should be rejected as stale
        assert!(registry.is_empty());
    }

    #[test]
    fn test_peer_registry_all_peers() {
        let registry = PeerRegistry::new();
        registry.upsert(make_test_peer("peer-A", 4, 5.0, 8.0, 500.0));
        registry.upsert(make_test_peer("peer-B", 8, 10.0, 16.0, 1000.0));
        registry.upsert(make_test_peer("peer-C", 2, 0.0, 4.0, 100.0));

        let all = registry.all_peers();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_get_best_peer_for_gpu_task() {
        let registry = PeerRegistry::new();
        // Peer A: low GPU
        registry.upsert(make_test_peer("peer-A", 8, 1.0, 16.0, 1000.0));
        // Peer B: high GPU
        registry.upsert(make_test_peer("peer-B", 4, 20.0, 8.0, 1000.0));
        // Peer C: no GPU
        registry.upsert(make_test_peer("peer-C", 16, 0.0, 32.0, 1000.0));

        let best = registry.get_best_peer_for_task("gpu-compute");
        assert!(best.is_some());
        // Peer B should win for GPU tasks due to highest gpu_tflops
        assert_eq!(best.unwrap().peer_id, "peer-B");
    }

    #[test]
    fn test_get_best_peer_for_zk_task() {
        let registry = PeerRegistry::new();
        // Peer A: many available cores
        registry.upsert(make_test_peer("peer-A", 16, 0.0, 8.0, 1000.0));
        // Peer B: fewer cores but has GPU
        registry.upsert(make_test_peer("peer-B", 4, 20.0, 32.0, 1000.0));

        let best = registry.get_best_peer_for_task("zk-proof");
        assert!(best.is_some());
        // Peer A should win for ZK tasks — CPU-bound, more cores
        assert_eq!(best.unwrap().peer_id, "peer-A");
    }

    #[test]
    fn test_get_best_peer_empty_registry() {
        let registry = PeerRegistry::new();
        assert!(registry.get_best_peer_for_task("mining").is_none());
    }

    #[test]
    fn test_get_best_peer_no_available_capacity() {
        let registry = PeerRegistry::new();
        // Peer with 0 available cores and no GPU
        registry.upsert(make_test_peer("busy-peer", 0, 0.0, 0.0, 1000.0));

        // Should return None — no capacity
        assert!(registry.get_best_peer_for_task("generic").is_none());
    }

    #[test]
    fn test_peer_score_function() {
        let peer = make_test_peer("test", 8, 10.0, 16.0, 1000.0);
        let score = compute_peer_score(&peer, "generic");
        assert!(score > 0.0, "Score should be positive for capable peer");

        // Zero-cores peer should have low score
        let zero_peer = make_test_peer("zero", 0, 0.0, 0.0, 1.0);
        let zero_score = compute_peer_score(&zero_peer, "generic");
        assert_eq!(zero_score, 0.0, "Score should be 0 for peer with no resources");
    }

    #[test]
    fn test_roundtrip_serialize_deserialize() {
        let snap = make_test_snapshot();
        let info = create_peer_announcement(&snap, "nuke", "12D3KooWRoundTrip");

        // Serialize -> bytes -> deserialize
        let bytes = serde_json::to_vec(&info).unwrap();
        let parsed = parse_peer_announcement(&bytes).unwrap();

        assert_eq!(parsed.peer_id, info.peer_id);
        assert_eq!(parsed.available_cores, info.available_cores);
        assert_eq!(parsed.total_cores, info.total_cores);
        assert!((parsed.gpu_tflops - info.gpu_tflops).abs() < 0.001);
        assert!((parsed.ram_available_gb - info.ram_available_gb).abs() < 0.01);
        assert_eq!(parsed.compute_mode, info.compute_mode);
        assert_eq!(parsed.version, info.version);
        assert_eq!(parsed.timestamp, info.timestamp);
    }

    // ═══════════════════════════════════════════════════════════════
    // Handshake protocol tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_handshake_creation() {
        let hs = TunnelHandshake::new(
            "initiator-A",
            "responder-B",
            vec!["mining".into(), "inference".into()],
        );
        assert_eq!(hs.initiator_peer_id, "initiator-A");
        assert_eq!(hs.responder_peer_id, "responder-B");
        assert_eq!(hs.capabilities.len(), 2);
        assert_eq!(hs.protocol_version, 1);
        assert!(hs.timestamp > 0);
        assert!(hs.nonce.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_handshake_serialization_roundtrip() {
        let hs = TunnelHandshake::new("peer-X", "peer-Y", vec!["mining".into(), "bridge-verify".into()]);
        let bytes = serde_json::to_vec(&hs).unwrap();
        let parsed: TunnelHandshake = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.initiator_peer_id, hs.initiator_peer_id);
        assert_eq!(parsed.responder_peer_id, hs.responder_peer_id);
        assert_eq!(parsed.nonce, hs.nonce);
        assert_eq!(parsed.capabilities, hs.capabilities);
        assert_eq!(parsed.protocol_version, hs.protocol_version);
        assert_eq!(parsed.timestamp, hs.timestamp);
    }

    #[test]
    fn test_handshake_response_accept() {
        let resp = HandshakeResponse::accept(vec!["mining".into()]);
        assert!(resp.accepted);
        assert!(resp.reason.is_none());
        assert_eq!(resp.capabilities, vec!["mining"]);
        assert!(resp.nonce.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_handshake_response_reject() {
        let resp = HandshakeResponse::reject("capacity full");
        assert!(!resp.accepted);
        assert_eq!(resp.reason.as_deref(), Some("capacity full"));
        assert!(resp.capabilities.is_empty());
    }

    #[test]
    fn test_handshake_response_serialization_roundtrip() {
        let resp = HandshakeResponse::accept(vec!["inference".into(), "mining".into()]);
        let bytes = serde_json::to_vec(&resp).unwrap();
        let parsed: HandshakeResponse = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.accepted, resp.accepted);
        assert_eq!(parsed.nonce, resp.nonce);
        assert_eq!(parsed.capabilities, resp.capabilities);
    }

    // ═══════════════════════════════════════════════════════════════
    // Tunnel state machine tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_tunnel_state_default_is_active() {
        let tunnel = ComputeTunnel::new("peer-1".into(), TunnelType::NodeToNode);
        assert_eq!(*tunnel.state.read(), TunnelState::Active);
        assert!(tunnel.is_active());
    }

    #[test]
    fn test_tunnel_with_handshaking_state() {
        let tunnel = ComputeTunnel::with_state("peer-2".into(), TunnelType::NodeToNode, TunnelState::Handshaking);
        assert_eq!(*tunnel.state.read(), TunnelState::Handshaking);
        assert!(tunnel.is_active());
    }

    #[test]
    fn test_tunnel_state_transitions_valid() {
        let tunnel = ComputeTunnel::with_state("peer-3".into(), TunnelType::NodeToNode, TunnelState::Handshaking);
        assert!(tunnel.transition_to(TunnelState::Active));
        assert_eq!(*tunnel.state.read(), TunnelState::Active);
        assert!(tunnel.transition_to(TunnelState::Draining));
        assert_eq!(*tunnel.state.read(), TunnelState::Draining);
        assert!(tunnel.transition_to(TunnelState::Closed));
        assert_eq!(*tunnel.state.read(), TunnelState::Closed);
        assert!(!tunnel.is_active());
    }

    #[test]
    fn test_tunnel_state_transitions_invalid() {
        let tunnel = ComputeTunnel::with_state("peer-4".into(), TunnelType::NodeToNode, TunnelState::Active);
        assert!(!tunnel.transition_to(TunnelState::Handshaking));
        assert_eq!(*tunnel.state.read(), TunnelState::Active);
        assert!(!tunnel.transition_to(TunnelState::Active));
    }

    #[test]
    fn test_tunnel_handshaking_to_closed() {
        let tunnel = ComputeTunnel::with_state("peer-5".into(), TunnelType::NodeToNode, TunnelState::Handshaking);
        assert!(tunnel.transition_to(TunnelState::Closed));
        assert!(!tunnel.is_active());
    }

    // ═══════════════════════════════════════════════════════════════
    // Multiplexed stream tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_open_stream_on_active_tunnel() {
        let tunnel = ComputeTunnel::new("peer-s1".into(), TunnelType::NodeToNode);
        let stream_id = tunnel.open_stream(StreamType::Mining);
        assert!(stream_id.is_ok());
        assert_eq!(tunnel.stream_count(), 1);
        let id2 = tunnel.open_stream(StreamType::Inference).unwrap();
        assert_eq!(tunnel.stream_count(), 2);
        assert_ne!(stream_id.unwrap(), id2);
    }

    #[test]
    fn test_cannot_open_stream_on_handshaking_tunnel() {
        let tunnel = ComputeTunnel::with_state("peer-s2".into(), TunnelType::NodeToNode, TunnelState::Handshaking);
        let result = tunnel.open_stream(StreamType::Mining);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Handshaking"));
    }

    #[test]
    fn test_stream_limit_enforced() {
        let tunnel = ComputeTunnel::new("peer-s3".into(), TunnelType::NodeToNode);
        for i in 0..MAX_STREAMS_PER_TUNNEL {
            let result = tunnel.open_stream(StreamType::Mining);
            assert!(result.is_ok(), "Failed to open stream {}", i);
        }
        let result = tunnel.open_stream(StreamType::Control);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Stream limit"));
    }

    #[test]
    fn test_close_stream() {
        let tunnel = ComputeTunnel::new("peer-s4".into(), TunnelType::NodeToNode);
        let id = tunnel.open_stream(StreamType::Inference).unwrap();
        assert_eq!(tunnel.stream_count(), 1);
        assert!(tunnel.close_stream(id));
        assert_eq!(tunnel.stream_count(), 0);
        assert!(!tunnel.close_stream(id));
    }

    #[test]
    fn test_close_stream_frees_slot() {
        let tunnel = ComputeTunnel::new("peer-s5".into(), TunnelType::NodeToNode);
        let mut ids = Vec::new();
        for _ in 0..MAX_STREAMS_PER_TUNNEL {
            ids.push(tunnel.open_stream(StreamType::Mining).unwrap());
        }
        assert!(tunnel.open_stream(StreamType::Control).is_err());
        tunnel.close_stream(ids[0]);
        assert!(tunnel.open_stream(StreamType::Control).is_ok());
    }

    #[test]
    fn test_stream_sender_retrieval() {
        let tunnel = ComputeTunnel::new("peer-s6".into(), TunnelType::NodeToNode);
        let id = tunnel.open_stream(StreamType::TensorShard).unwrap();
        assert!(tunnel.stream_sender(id).is_some());
        assert!(tunnel.stream_sender(9999).is_none());
    }

    #[test]
    fn test_take_receiver_once() {
        let tunnel = ComputeTunnel::new("peer-s7".into(), TunnelType::NodeToNode);
        assert!(tunnel.take_receiver().is_some());
        assert!(tunnel.take_receiver().is_none());
    }

    #[test]
    fn test_stream_type_display() {
        assert_eq!(format!("{}", StreamType::Mining), "mining");
        assert_eq!(format!("{}", StreamType::Inference), "inference");
        assert_eq!(format!("{}", StreamType::BridgeVerify), "bridge-verify");
        assert_eq!(format!("{}", StreamType::TensorShard), "tensor-shard");
        assert_eq!(format!("{}", StreamType::Control), "control");
    }

    // ═══════════════════════════════════════════════════════════════
    // Tunnel manager handshake tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_open_tunnel_with_handshake() {
        let mgr = TunnelManager::with_identity(10, "local-peer".into(), vec!["mining".into(), "inference".into()]);
        let result = mgr.open_tunnel_with_handshake("remote-1", TunnelType::NodeToNode);
        assert!(result.is_ok());
        let hs = result.unwrap();
        assert_eq!(hs.initiator_peer_id, "local-peer");
        assert_eq!(hs.responder_peer_id, "remote-1");
        assert_eq!(hs.capabilities, vec!["mining", "inference"]);
        assert_eq!(mgr.tunnel_state("remote-1"), Some(TunnelState::Handshaking));
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_open_tunnel_with_handshake_duplicate_rejected() {
        let mgr = TunnelManager::with_identity(10, "local".into(), vec![]);
        mgr.open_tunnel_with_handshake("remote-1", TunnelType::NodeToNode).unwrap();
        let result = mgr.open_tunnel_with_handshake("remote-1", TunnelType::NodeToNode);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Already have tunnel"));
    }

    #[test]
    fn test_open_tunnel_with_handshake_max_limit() {
        let mgr = TunnelManager::with_identity(1, "local".into(), vec![]);
        mgr.open_tunnel_with_handshake("remote-1", TunnelType::NodeToNode).unwrap();
        let result = mgr.open_tunnel_with_handshake("remote-2", TunnelType::NodeToNode);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Max tunnels reached"));
    }

    #[test]
    fn test_complete_handshake_accepted() {
        let mgr = TunnelManager::with_identity(10, "local".into(), vec!["mining".into(), "inference".into()]);
        mgr.open_tunnel_with_handshake("remote-1", TunnelType::NodeToNode).unwrap();
        let response = HandshakeResponse::accept(vec!["mining".into()]);
        let result = mgr.complete_handshake("remote-1", &response);
        assert!(result.is_ok());
        assert_eq!(mgr.tunnel_state("remote-1"), Some(TunnelState::Active));
        assert_eq!(mgr.active_count(), 1);
        assert_eq!(mgr.tunnel_capabilities("remote-1"), Some(vec!["mining".to_string()]));
    }

    #[test]
    fn test_complete_handshake_rejected() {
        let mgr = TunnelManager::with_identity(10, "local".into(), vec![]);
        mgr.open_tunnel_with_handshake("remote-1", TunnelType::NodeToNode).unwrap();
        let response = HandshakeResponse::reject("no capacity");
        let result = mgr.complete_handshake("remote-1", &response);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Handshake rejected"));
        assert_eq!(mgr.tunnel_state("remote-1"), None);
    }

    #[test]
    fn test_complete_handshake_no_pending_tunnel() {
        let mgr = TunnelManager::with_identity(10, "local".into(), vec![]);
        let response = HandshakeResponse::accept(vec![]);
        let result = mgr.complete_handshake("nonexistent", &response);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No pending tunnel"));
    }

    #[test]
    fn test_complete_handshake_wrong_state() {
        let mgr = TunnelManager::with_identity(10, "local".into(), vec![]);
        mgr.open_tunnel("remote-1", TunnelType::NodeToNode);
        let response = HandshakeResponse::accept(vec![]);
        let result = mgr.complete_handshake("remote-1", &response);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not Handshaking"));
    }

    #[test]
    fn test_handle_incoming_handshake_accept() {
        let mgr = TunnelManager::with_identity(10, "responder-local".into(), vec!["mining".into(), "inference".into(), "bridge-verify".into()]);
        let hs = TunnelHandshake::new("remote-initiator", "responder-local", vec!["mining".into(), "tensor-shard".into()]);
        let resp = mgr.handle_incoming_handshake(&hs);
        assert!(resp.accepted);
        assert_eq!(resp.capabilities, vec!["mining"]);
        assert_eq!(mgr.tunnel_state("remote-initiator"), Some(TunnelState::Active));
    }

    #[test]
    fn test_handle_incoming_handshake_reject_capacity() {
        let mgr = TunnelManager::with_identity(1, "local".into(), vec!["mining".into()]);
        mgr.open_tunnel("existing-peer", TunnelType::NodeToNode);
        let hs = TunnelHandshake::new("new-peer", "local", vec!["mining".into()]);
        let resp = mgr.handle_incoming_handshake(&hs);
        assert!(!resp.accepted);
        assert!(resp.reason.as_deref().unwrap().contains("capacity full"));
    }

    #[test]
    fn test_handle_incoming_handshake_reject_bad_version() {
        let mgr = TunnelManager::with_identity(10, "local".into(), vec![]);
        let mut hs = TunnelHandshake::new("remote", "local", vec![]);
        hs.protocol_version = 99;
        let resp = mgr.handle_incoming_handshake(&hs);
        assert!(!resp.accepted);
        assert!(resp.reason.as_deref().unwrap().contains("protocol version"));
    }

    #[test]
    fn test_handle_incoming_handshake_reject_stale() {
        let mgr = TunnelManager::with_identity(10, "local".into(), vec![]);
        let mut hs = TunnelHandshake::new("remote", "local", vec![]);
        hs.timestamp = 1000;
        let resp = mgr.handle_incoming_handshake(&hs);
        assert!(!resp.accepted);
        assert!(resp.reason.as_deref().unwrap().contains("too old"));
    }

    #[test]
    fn test_handle_incoming_handshake_reject_duplicate() {
        let mgr = TunnelManager::with_identity(10, "local".into(), vec!["mining".into()]);
        let hs = TunnelHandshake::new("remote", "local", vec!["mining".into()]);
        let resp1 = mgr.handle_incoming_handshake(&hs);
        assert!(resp1.accepted);
        let resp2 = mgr.handle_incoming_handshake(&hs);
        assert!(!resp2.accepted);
        assert!(resp2.reason.as_deref().unwrap().contains("already exists"));
    }

    // ═══════════════════════════════════════════════════════════════
    // Drain and lifecycle tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_drain_tunnel() {
        let mgr = TunnelManager::with_identity(10, "local".into(), vec![]);
        mgr.open_tunnel("peer-drain", TunnelType::NodeToNode);
        assert_eq!(mgr.active_count(), 1);
        assert!(mgr.drain_tunnel("peer-drain"));
        assert_eq!(mgr.active_count(), 0);
        assert_eq!(mgr.tunnel_state("peer-drain"), Some(TunnelState::Draining));
    }

    #[test]
    fn test_drain_nonexistent_tunnel() {
        let mgr = TunnelManager::new(10);
        assert!(!mgr.drain_tunnel("no-such-peer"));
    }

    #[test]
    fn test_route_work_skips_non_active_tunnels() {
        let mgr = TunnelManager::with_identity(10, "local".into(), vec![]);
        mgr.open_tunnel_with_handshake("peer-hs", TunnelType::NodeToNode).unwrap();
        let item = TunnelWorkItem { id: 42, layer: ComputeLayer::Mining, payload_bytes: 128, priority: 0, sender_peer: "local".to_string() };
        assert_eq!(mgr.route_work(&item), None);
        mgr.open_tunnel("peer-active", TunnelType::NodeToNode);
        let routed = mgr.route_work(&item);
        assert_eq!(routed, Some("peer-active".to_string()));
    }

    // ═══════════════════════════════════════════════════════════════
    // Manager stream delegation tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_manager_open_stream_on_tunnel() {
        let mgr = TunnelManager::new(10);
        mgr.open_tunnel("peer-ms1", TunnelType::NodeToNode);
        let result = mgr.open_stream_on_tunnel("peer-ms1", StreamType::Mining);
        assert!(result.is_ok());
        let result2 = mgr.open_stream_on_tunnel("ghost-peer", StreamType::Mining);
        assert!(result2.is_err());
    }

    #[test]
    fn test_manager_close_stream_on_tunnel() {
        let mgr = TunnelManager::new(10);
        mgr.open_tunnel("peer-ms2", TunnelType::NodeToNode);
        let stream_id = mgr.open_stream_on_tunnel("peer-ms2", StreamType::Inference).unwrap();
        assert!(mgr.close_stream_on_tunnel("peer-ms2", stream_id));
        assert!(!mgr.close_stream_on_tunnel("peer-ms2", stream_id));
        assert!(!mgr.close_stream_on_tunnel("ghost", 1));
    }

    // ═══════════════════════════════════════════════════════════════
    // Auto-connect tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_auto_connect_to_best_peers() {
        let mgr = TunnelManager::with_identity(10, "local-node".into(), vec!["mining".into()]);
        let registry = mgr.peer_registry();
        registry.upsert(make_test_peer("peer-A", 8, 10.0, 16.0, 1000.0));
        registry.upsert(make_test_peer("peer-B", 4, 5.0, 8.0, 500.0));
        registry.upsert(make_test_peer("peer-C", 16, 20.0, 32.0, 2000.0));
        let opened = mgr.auto_connect_to_best_peers(0.01, 5);
        assert_eq!(opened.len(), 2);
        for peer_id in &opened {
            assert_eq!(mgr.tunnel_state(peer_id), Some(TunnelState::Handshaking));
        }
    }

    #[test]
    fn test_auto_connect_skips_existing_tunnels() {
        let mgr = TunnelManager::with_identity(10, "local".into(), vec![]);
        let registry = mgr.peer_registry();
        registry.upsert(make_test_peer("peer-A", 8, 10.0, 16.0, 1000.0));
        registry.upsert(make_test_peer("peer-B", 4, 5.0, 8.0, 500.0));
        mgr.open_tunnel("peer-A", TunnelType::NodeToNode);
        let opened = mgr.auto_connect_to_best_peers(0.01, 5);
        assert_eq!(opened.len(), 1);
        assert_eq!(opened[0], "peer-B");
    }

    #[test]
    fn test_auto_connect_skips_self() {
        let mgr = TunnelManager::with_identity(10, "self-node".into(), vec![]);
        let registry = mgr.peer_registry();
        registry.upsert(make_test_peer("self-node", 8, 10.0, 16.0, 1000.0));
        registry.upsert(make_test_peer("other-node", 4, 5.0, 8.0, 500.0));
        let opened = mgr.auto_connect_to_best_peers(0.01, 5);
        assert_eq!(opened.len(), 1);
        assert_eq!(opened[0], "other-node");
    }

    #[test]
    fn test_auto_connect_respects_min_score() {
        let mgr = TunnelManager::with_identity(10, "local".into(), vec![]);
        let registry = mgr.peer_registry();
        registry.upsert(make_test_peer("low-peer", 0, 0.0, 0.0, 1.0));
        let opened = mgr.auto_connect_to_best_peers(1.0, 5);
        assert!(opened.is_empty());
    }

    #[test]
    fn test_auto_connect_empty_registry() {
        let mgr = TunnelManager::with_identity(10, "local".into(), vec![]);
        let opened = mgr.auto_connect_to_best_peers(0.01, 5);
        assert!(opened.is_empty());
    }

    // ═══════════════════════════════════════════════════════════════
    // Full handshake flow (initiator + responder)
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_full_handshake_flow() {
        let initiator = TunnelManager::with_identity(10, "node-A".into(), vec!["mining".into(), "inference".into()]);
        let responder = TunnelManager::with_identity(10, "node-B".into(), vec!["mining".into(), "bridge-verify".into()]);
        let handshake = initiator.open_tunnel_with_handshake("node-B", TunnelType::NodeToNode).unwrap();
        assert_eq!(initiator.tunnel_state("node-B"), Some(TunnelState::Handshaking));
        let response = responder.handle_incoming_handshake(&handshake);
        assert!(response.accepted);
        assert_eq!(response.capabilities, vec!["mining"]);
        assert_eq!(responder.tunnel_state("node-A"), Some(TunnelState::Active));
        let result = initiator.complete_handshake("node-B", &response);
        assert!(result.is_ok());
        assert_eq!(initiator.tunnel_state("node-B"), Some(TunnelState::Active));
        assert_eq!(initiator.active_count(), 1);
        assert_eq!(responder.active_count(), 1);
        assert_eq!(initiator.tunnel_capabilities("node-B"), Some(vec!["mining".to_string()]));
        assert_eq!(responder.tunnel_capabilities("node-A"), Some(vec!["mining".to_string()]));
    }

    #[test]
    fn test_full_handshake_flow_with_streams() {
        let initiator = TunnelManager::with_identity(10, "node-A".into(), vec!["mining".into()]);
        let responder = TunnelManager::with_identity(10, "node-B".into(), vec!["mining".into()]);
        let hs = initiator.open_tunnel_with_handshake("node-B", TunnelType::NodeToNode).unwrap();
        let resp = responder.handle_incoming_handshake(&hs);
        initiator.complete_handshake("node-B", &resp).unwrap();
        let s1 = initiator.open_stream_on_tunnel("node-B", StreamType::Mining);
        assert!(s1.is_ok());
        let s2 = initiator.open_stream_on_tunnel("node-B", StreamType::Control);
        assert!(s2.is_ok());
        let s3 = responder.open_stream_on_tunnel("node-A", StreamType::Mining);
        assert!(s3.is_ok());
    }

    // ═══════════════════════════════════════════════════════════════
    // Crypto Handshake tests (Issue #002 — Criterion 4)
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_crypto_handshake_init_state() {
        let hs = CryptoHandshake::new("peer-A");
        assert_eq!(hs.state(), HandshakeState::Init);
        assert!(hs.session_key().is_none());
    }

    #[test]
    fn test_crypto_handshake_initiate() {
        let mut hs = CryptoHandshake::new("peer-A");
        let init = hs.initiate_handshake("peer-B");
        assert!(init.is_ok());
        let init = init.unwrap();
        assert_eq!(init.initiator_peer_id, "peer-A");
        assert!(init.timestamp > 0);
        assert!(init.ephemeral_public_key.iter().any(|&b| b != 0));
        assert_eq!(hs.state(), HandshakeState::KeyExchangeSent);
    }

    #[test]
    fn test_crypto_handshake_cannot_initiate_twice() {
        let mut hs = CryptoHandshake::new("peer-A");
        hs.initiate_handshake("peer-B").unwrap();
        let result = hs.initiate_handshake("peer-C");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("KeyExchangeSent"));
    }

    #[test]
    fn test_crypto_handshake_full_exchange() {
        // Initiator side
        let mut initiator = CryptoHandshake::new("alice");
        let init = initiator.initiate_handshake("bob").unwrap();
        assert_eq!(initiator.state(), HandshakeState::KeyExchangeSent);

        // Responder side
        let mut responder = CryptoHandshake::new("bob");
        let (response, responder_key) = responder.respond_to_handshake(&init).unwrap();
        assert_eq!(responder.state(), HandshakeState::Established);

        // Initiator completes
        let initiator_key = initiator.complete_handshake(&init, &response).unwrap();
        assert_eq!(initiator.state(), HandshakeState::Established);

        // Both sides must derive the SAME session key
        assert_eq!(initiator_key.as_bytes(), responder_key.as_bytes());
    }

    #[test]
    fn test_crypto_handshake_different_pairs_different_keys() {
        // Pair 1
        let mut init1 = CryptoHandshake::new("a1");
        let init_msg1 = init1.initiate_handshake("b1").unwrap();
        let mut resp1 = CryptoHandshake::new("b1");
        let (resp_msg1, key1_resp) = resp1.respond_to_handshake(&init_msg1).unwrap();
        let key1_init = init1.complete_handshake(&init_msg1, &resp_msg1).unwrap();

        // Pair 2
        let mut init2 = CryptoHandshake::new("a2");
        let init_msg2 = init2.initiate_handshake("b2").unwrap();
        let mut resp2 = CryptoHandshake::new("b2");
        let (resp_msg2, key2_resp) = resp2.respond_to_handshake(&init_msg2).unwrap();
        let key2_init = init2.complete_handshake(&init_msg2, &resp_msg2).unwrap();

        // Keys within each pair match
        assert_eq!(key1_init.as_bytes(), key1_resp.as_bytes());
        assert_eq!(key2_init.as_bytes(), key2_resp.as_bytes());

        // Keys across pairs differ (different ephemeral keys)
        assert_ne!(key1_init.as_bytes(), key2_init.as_bytes());
    }

    #[test]
    fn test_crypto_handshake_nonce_mismatch_rejected() {
        let mut initiator = CryptoHandshake::new("alice");
        let init = initiator.initiate_handshake("bob").unwrap();

        let mut responder = CryptoHandshake::new("bob");
        let (mut response, _) = responder.respond_to_handshake(&init).unwrap();

        // Tamper with the echoed nonce
        response.initiator_nonce[0] ^= 0xFF;

        let result = initiator.complete_handshake(&init, &response);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Nonce mismatch"));
    }

    #[test]
    fn test_crypto_handshake_cannot_complete_from_init() {
        let mut hs = CryptoHandshake::new("peer-A");
        let fake_response = HandshakeKeyResponse {
            responder_peer_id: "peer-B".into(),
            ephemeral_public_key: [0u8; 32],
            initiator_nonce: [0u8; 32],
            nonce: [0u8; 32],
            timestamp: 0,
        };
        let fake_init = HandshakeInit {
            initiator_peer_id: "peer-A".into(),
            ephemeral_public_key: [0u8; 32],
            nonce: [0u8; 32],
            timestamp: 0,
        };
        let result = hs.complete_handshake(&fake_init, &fake_response);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Init"));
    }

    #[test]
    fn test_crypto_handshake_respond_not_from_init() {
        let mut hs = CryptoHandshake::new("peer-A");
        let init_msg = hs.initiate_handshake("peer-B").unwrap();
        // Now in KeyExchangeSent — cannot respond
        let result = hs.respond_to_handshake(&init_msg);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("KeyExchangeSent"));
    }

    #[test]
    fn test_handshake_state_display() {
        assert_eq!(format!("{}", HandshakeState::Init), "Init");
        assert_eq!(format!("{}", HandshakeState::KeyExchangeSent), "KeyExchangeSent");
        assert_eq!(format!("{}", HandshakeState::KeyExchangeReceived), "KeyExchangeReceived");
        assert_eq!(format!("{}", HandshakeState::Established), "Established");
    }

    #[test]
    fn test_session_key_debug_redacted() {
        let key = SessionKey { key: [42u8; 32] };
        let debug_str = format!("{:?}", key);
        assert!(debug_str.contains("REDACTED"));
        assert!(!debug_str.contains("42"));
    }

    #[test]
    fn test_handshake_init_serialization_roundtrip() {
        let mut hs = CryptoHandshake::new("peer-A");
        let init = hs.initiate_handshake("peer-B").unwrap();
        let bytes = serde_json::to_vec(&init).unwrap();
        let parsed: HandshakeInit = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.initiator_peer_id, init.initiator_peer_id);
        assert_eq!(parsed.ephemeral_public_key, init.ephemeral_public_key);
        assert_eq!(parsed.nonce, init.nonce);
        assert_eq!(parsed.timestamp, init.timestamp);
    }

    #[test]
    fn test_handshake_key_response_serialization_roundtrip() {
        let mut initiator = CryptoHandshake::new("alice");
        let init = initiator.initiate_handshake("bob").unwrap();
        let mut responder = CryptoHandshake::new("bob");
        let (response, _) = responder.respond_to_handshake(&init).unwrap();

        let bytes = serde_json::to_vec(&response).unwrap();
        let parsed: HandshakeKeyResponse = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.responder_peer_id, response.responder_peer_id);
        assert_eq!(parsed.ephemeral_public_key, response.ephemeral_public_key);
        assert_eq!(parsed.initiator_nonce, response.initiator_nonce);
        assert_eq!(parsed.nonce, response.nonce);
    }

    // ═══════════════════════════════════════════════════════════════
    // Frame encoding/decoding tests (Issue #002 — Criterion 5)
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_encode_decode_frame() {
        let payload = b"hello world";
        let stream_id = 42u32;
        let frame = encode_frame(stream_id, payload);

        assert_eq!(frame.len(), FRAME_HEADER_SIZE + payload.len());

        let decoded = decode_frame(&frame);
        assert!(decoded.is_some());
        let (dec_id, dec_payload, consumed) = decoded.unwrap();
        assert_eq!(dec_id, stream_id);
        assert_eq!(dec_payload, payload);
        assert_eq!(consumed, frame.len());
    }

    #[test]
    fn test_encode_decode_empty_payload() {
        let frame = encode_frame(1, b"");
        let decoded = decode_frame(&frame);
        assert!(decoded.is_some());
        let (id, payload, consumed) = decoded.unwrap();
        assert_eq!(id, 1);
        assert!(payload.is_empty());
        assert_eq!(consumed, FRAME_HEADER_SIZE);
    }

    #[test]
    fn test_decode_frame_too_short() {
        assert!(decode_frame(&[0u8; 3]).is_none()); // Less than header
        assert!(decode_frame(&[]).is_none());        // Empty
    }

    #[test]
    fn test_decode_frame_incomplete_payload() {
        let mut frame = encode_frame(1, b"full payload");
        frame.truncate(FRAME_HEADER_SIZE + 2); // Truncate payload
        assert!(decode_frame(&frame).is_none());
    }

    #[test]
    fn test_encode_decode_multiple_frames_concatenated() {
        let frame1 = encode_frame(1, b"first");
        let frame2 = encode_frame(2, b"second");
        let frame3 = encode_frame(3, b"third");

        let mut combined = Vec::new();
        combined.extend_from_slice(&frame1);
        combined.extend_from_slice(&frame2);
        combined.extend_from_slice(&frame3);

        let mut offset = 0;
        let mut decoded = Vec::new();
        while offset < combined.len() {
            if let Some((id, payload, consumed)) = decode_frame(&combined[offset..]) {
                decoded.push((id, payload));
                offset += consumed;
            } else {
                break;
            }
        }

        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].0, 1);
        assert_eq!(decoded[0].1, b"first");
        assert_eq!(decoded[1].0, 2);
        assert_eq!(decoded[1].1, b"second");
        assert_eq!(decoded[2].0, 3);
        assert_eq!(decoded[2].1, b"third");
    }

    #[test]
    fn test_frame_stream_type_byte_roundtrip() {
        for st in [
            FrameStreamType::Mining,
            FrameStreamType::Inference,
            FrameStreamType::Proof,
            FrameStreamType::Control,
        ] {
            let b = st.to_byte();
            let decoded = FrameStreamType::from_byte(b);
            assert_eq!(decoded, Some(st));
        }
        assert!(FrameStreamType::from_byte(99).is_none());
    }

    #[test]
    fn test_frame_stream_type_display() {
        assert_eq!(format!("{}", FrameStreamType::Mining), "mining");
        assert_eq!(format!("{}", FrameStreamType::Inference), "inference");
        assert_eq!(format!("{}", FrameStreamType::Proof), "proof");
        assert_eq!(format!("{}", FrameStreamType::Control), "control");
    }

    // ═══════════════════════════════════════════════════════════════
    // FramedTunnelStream integration tests (Issue #002 — Criterion 5)
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_framed_stream_open_and_close() {
        let mut framed = FramedTunnelStream::new();
        assert_eq!(framed.stream_count(), 0);

        let h1 = framed.open_stream(FrameStreamType::Mining);
        assert_eq!(h1.stream_id, 1);
        assert_eq!(framed.stream_count(), 1);

        let h2 = framed.open_stream(FrameStreamType::Control);
        assert_eq!(h2.stream_id, 2);
        assert_eq!(framed.stream_count(), 2);

        framed.close_stream(&h1);
        assert_eq!(framed.stream_count(), 1);

        framed.close_stream(&h2);
        assert_eq!(framed.stream_count(), 0);
    }

    #[test]
    fn test_framed_stream_send_and_recv() {
        let mut framed = FramedTunnelStream::new();
        let h = framed.open_stream(FrameStreamType::Mining);

        // Send data
        framed.send_on_stream(&h, b"test payload").unwrap();
        assert_eq!(framed.outbound_pending(), 1);

        // Flush to wire bytes
        let wire = framed.flush_outbound();
        assert_eq!(framed.outbound_pending(), 0);
        assert!(!wire.is_empty());

        // Simulate receiving those bytes
        let count = framed.feed_incoming(&wire);
        assert_eq!(count, 1);

        // Receive on the stream
        let data = framed.recv_on_stream(&h).unwrap();
        assert_eq!(data, b"test payload");
    }

    #[test]
    fn test_framed_stream_multiple_streams() {
        let mut framed = FramedTunnelStream::new();
        let mining = framed.open_stream(FrameStreamType::Mining);
        let control = framed.open_stream(FrameStreamType::Control);

        framed.send_on_stream(&mining, b"mine-data").unwrap();
        framed.send_on_stream(&control, b"ctrl-msg").unwrap();

        let wire = framed.flush_outbound();
        let count = framed.feed_incoming(&wire);
        assert_eq!(count, 2);

        let mine_data = framed.recv_on_stream(&mining).unwrap();
        assert_eq!(mine_data, b"mine-data");

        let ctrl_data = framed.recv_on_stream(&control).unwrap();
        assert_eq!(ctrl_data, b"ctrl-msg");
    }

    #[test]
    fn test_framed_stream_send_on_closed_stream() {
        let mut framed = FramedTunnelStream::new();
        let h = framed.open_stream(FrameStreamType::Proof);
        framed.close_stream(&h);
        let result = framed.send_on_stream(&h, b"data");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_framed_stream_recv_empty() {
        let mut framed = FramedTunnelStream::new();
        let h = framed.open_stream(FrameStreamType::Inference);
        let result = framed.recv_on_stream(&h);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No data"));
    }

    #[test]
    fn test_framed_stream_payload_too_large() {
        let mut framed = FramedTunnelStream::new();
        let h = framed.open_stream(FrameStreamType::Mining);
        let big = vec![0u8; MAX_FRAME_PAYLOAD + 1];
        let result = framed.send_on_stream(&h, &big);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too large"));
    }

    #[test]
    fn test_framed_stream_feed_unknown_stream_drops() {
        let mut framed = FramedTunnelStream::new();
        // Create a frame for stream_id=99 which does not exist
        let frame = encode_frame(99, b"orphan");
        let count = framed.feed_incoming(&frame);
        // Frame is parsed but dropped (no stream 99)
        assert_eq!(count, 1);
    }

    #[test]
    fn test_framed_stream_multiple_recv_fifo() {
        let mut framed = FramedTunnelStream::new();
        let h = framed.open_stream(FrameStreamType::Mining);

        framed.send_on_stream(&h, b"first").unwrap();
        framed.send_on_stream(&h, b"second").unwrap();
        framed.send_on_stream(&h, b"third").unwrap();

        let wire = framed.flush_outbound();
        framed.feed_incoming(&wire);

        assert_eq!(framed.recv_on_stream(&h).unwrap(), b"first");
        assert_eq!(framed.recv_on_stream(&h).unwrap(), b"second");
        assert_eq!(framed.recv_on_stream(&h).unwrap(), b"third");
        assert!(framed.recv_on_stream(&h).is_err());
    }

    // ═══════════════════════════════════════════════════════════════
    // Result verification tests (Issue #002 — Criterion 6)
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_verify_results_unanimous() {
        let results = vec![
            ("peer-A".to_string(), b"result-X".to_vec()),
            ("peer-B".to_string(), b"result-X".to_vec()),
            ("peer-C".to_string(), b"result-X".to_vec()),
        ];
        let vr = ResultVerifier::verify_results(results);
        assert!(vr.accepted);
        assert_eq!(vr.result, b"result-X");
        assert!((vr.agreement - 1.0).abs() < 0.001);
        assert_eq!(vr.participating_peers.len(), 3);
    }

    #[test]
    fn test_verify_results_two_of_three() {
        let results = vec![
            ("peer-A".to_string(), b"correct".to_vec()),
            ("peer-B".to_string(), b"correct".to_vec()),
            ("peer-C".to_string(), b"different".to_vec()),
        ];
        let vr = ResultVerifier::verify_results(results);
        assert!(vr.accepted);
        assert_eq!(vr.result, b"correct");
        assert!((vr.agreement - 2.0 / 3.0).abs() < 0.01);
        assert_eq!(vr.participating_peers.len(), 3);
    }

    #[test]
    fn test_verify_results_all_different() {
        let results = vec![
            ("peer-A".to_string(), b"result-1".to_vec()),
            ("peer-B".to_string(), b"result-2".to_vec()),
            ("peer-C".to_string(), b"result-3".to_vec()),
        ];
        let vr = ResultVerifier::verify_results(results);
        assert!(!vr.accepted);
        assert!(vr.result.is_empty());
        assert!((vr.agreement - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_verify_results_no_responses() {
        let results: Vec<(String, Vec<u8>)> = vec![];
        let vr = ResultVerifier::verify_results(results);
        assert!(!vr.accepted);
        assert!(vr.result.is_empty());
        assert_eq!(vr.agreement, 0.0);
        assert!(vr.participating_peers.is_empty());
    }

    #[test]
    fn test_verify_results_single_response() {
        let results = vec![("peer-A".to_string(), b"solo".to_vec())];
        let vr = ResultVerifier::verify_results(results);
        // 1 of 1 does not meet MIN_AGREEMENT (2)
        assert!(!vr.accepted);
        assert!(vr.result.is_empty());
        assert!((vr.agreement - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_verify_results_two_responses_both_agree() {
        let results = vec![
            ("peer-A".to_string(), b"match".to_vec()),
            ("peer-B".to_string(), b"match".to_vec()),
        ];
        let vr = ResultVerifier::verify_results(results);
        assert!(vr.accepted);
        assert_eq!(vr.result, b"match");
        assert!((vr.agreement - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_verify_results_two_responses_disagree() {
        let results = vec![
            ("peer-A".to_string(), b"val-1".to_vec()),
            ("peer-B".to_string(), b"val-2".to_vec()),
        ];
        let vr = ResultVerifier::verify_results(results);
        assert!(!vr.accepted);
        assert!(vr.result.is_empty());
        assert!((vr.agreement - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_verify_results_empty_payloads_match() {
        let results = vec![
            ("peer-A".to_string(), vec![]),
            ("peer-B".to_string(), vec![]),
            ("peer-C".to_string(), vec![]),
        ];
        let vr = ResultVerifier::verify_results(results);
        assert!(vr.accepted);
        assert!(vr.result.is_empty()); // empty is a valid unanimous result
        assert!((vr.agreement - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_submit_verified_task_async_unanimous() {
        let peers = vec![
            PeerInfo { peer_id: "p1".into() },
            PeerInfo { peer_id: "p2".into() },
            PeerInfo { peer_id: "p3".into() },
        ];
        let task = TunnelPayload::MiningSubmit(b"task-data".to_vec());

        let vr = ResultVerifier::submit_verified_task(&task, &peers, |_peer_id, _payload| async {
            Ok(b"unanimous-result".to_vec())
        })
        .await;

        assert!(vr.accepted);
        assert_eq!(vr.result, b"unanimous-result");
        assert_eq!(vr.participating_peers.len(), 3);
        assert!((vr.agreement - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_submit_verified_task_async_two_of_three() {
        let peers = vec![
            PeerInfo { peer_id: "p1".into() },
            PeerInfo { peer_id: "p2".into() },
            PeerInfo { peer_id: "p3".into() },
        ];
        let task = TunnelPayload::MiningSubmit(vec![]);

        let vr = ResultVerifier::submit_verified_task(&task, &peers, |peer_id, _payload| async move {
            if peer_id == "p3" {
                Ok(b"outlier".to_vec())
            } else {
                Ok(b"consensus".to_vec())
            }
        })
        .await;

        assert!(vr.accepted);
        assert_eq!(vr.result, b"consensus");
    }

    #[tokio::test]
    async fn test_submit_verified_task_async_with_error() {
        let peers = vec![
            PeerInfo { peer_id: "p1".into() },
            PeerInfo { peer_id: "p2".into() },
            PeerInfo { peer_id: "p3".into() },
        ];
        let task = TunnelPayload::MiningSubmit(vec![]);

        let vr = ResultVerifier::submit_verified_task(&task, &peers, |peer_id, _payload| async move {
            if peer_id == "p3" {
                Err("connection failed".to_string())
            } else {
                Ok(b"good-result".to_vec())
            }
        })
        .await;

        assert!(vr.accepted);
        assert_eq!(vr.result, b"good-result");
        assert_eq!(vr.participating_peers.len(), 2);
    }

    #[tokio::test]
    async fn test_submit_verified_task_async_no_peers() {
        let peers: Vec<PeerInfo> = vec![];
        let task = TunnelPayload::MiningSubmit(vec![]);

        let vr = ResultVerifier::submit_verified_task(&task, &peers, |_peer_id, _payload| async {
            Ok(vec![])
        })
        .await;

        assert!(!vr.accepted);
        assert!(vr.participating_peers.is_empty());
    }

    #[tokio::test]
    async fn test_submit_verified_task_async_all_errors() {
        let peers = vec![
            PeerInfo { peer_id: "p1".into() },
            PeerInfo { peer_id: "p2".into() },
        ];
        let task = TunnelPayload::MiningSubmit(vec![]);

        let vr = ResultVerifier::submit_verified_task(&task, &peers, |_peer_id, _payload| async {
            Err("all fail".to_string())
        })
        .await;

        assert!(!vr.accepted);
        assert!(vr.result.is_empty());
        assert!(vr.participating_peers.is_empty());
    }

    // ═══════════════════════════════════════════════════════════════
    // Issue #024: Rekey / Forward Secrecy tests
    // ═══════════════════════════════════════════════════════════════

    /// Helper: create a SessionKey from a test byte pattern.
    fn make_test_session_key(seed: u8) -> SessionKey {
        SessionKey { key: [seed; 32] }
    }

    #[test]
    fn test_rekey_manager_creation() {
        let key = make_test_session_key(0xAA);
        let mgr = RekeyManager::new(key);

        assert_eq!(mgr.state(), RekeyState::Idle);
        assert_eq!(mgr.rekey_count(), 0);
        assert_eq!(mgr.bytes_since_rekey(), 0);
        assert!(mgr.current_key().is_some());
        assert_eq!(mgr.current_key().unwrap().as_bytes(), &[0xAA; 32]);
    }

    #[test]
    fn test_rekey_manager_record_bytes() {
        let key = make_test_session_key(0x01);
        let mut mgr = RekeyManager::new(key);

        mgr.record_bytes(1000);
        assert_eq!(mgr.bytes_since_rekey(), 1000);

        mgr.record_bytes(5000);
        assert_eq!(mgr.bytes_since_rekey(), 6000);
    }

    #[test]
    fn test_rekey_should_not_trigger_immediately() {
        let key = make_test_session_key(0x02);
        let mgr = RekeyManager::new(key);

        // Just created — neither time nor byte threshold met
        assert!(!mgr.should_rekey());
    }

    #[test]
    fn test_rekey_triggers_on_byte_threshold() {
        let key = make_test_session_key(0x03);
        let mut mgr = RekeyManager::new(key);

        // Record just under the threshold
        mgr.record_bytes(REKEY_BYTE_THRESHOLD - 1);
        assert!(!mgr.should_rekey());

        // Push over the threshold
        mgr.record_bytes(1);
        assert!(mgr.should_rekey());
    }

    #[test]
    fn test_rekey_full_protocol_initiator_responder() {
        // Simulate a full rekey exchange between initiator and responder.
        let initial_key_i = make_test_session_key(0x10);
        let initial_key_r = make_test_session_key(0x10);

        let mut initiator = RekeyManager::new(initial_key_i);
        let mut responder = RekeyManager::new(initial_key_r);

        // Both start with the same initial key
        assert_eq!(
            initiator.current_key().unwrap().as_bytes(),
            responder.current_key().unwrap().as_bytes(),
        );

        // Step 1: Initiator generates rekey message
        let rekey_payload = initiator.initiate_rekey().unwrap();
        assert_eq!(initiator.state(), RekeyState::AwaitingAck);

        // Extract the pubkey and nonce from the payload
        let (their_pubkey, their_nonce, rekey_seq) = match &rekey_payload {
            TunnelPayload::Rekey { ephemeral_pubkey, nonce, rekey_seq } => {
                (*ephemeral_pubkey, *nonce, *rekey_seq)
            }
            _ => panic!("Expected TunnelPayload::Rekey"),
        };
        assert_eq!(rekey_seq, 1);

        // Step 2: Responder handles the rekey request
        let ack_payload = responder
            .handle_rekey_request(&their_pubkey, &their_nonce, rekey_seq)
            .unwrap();
        assert_eq!(responder.rekey_count(), 1);
        assert_eq!(responder.bytes_since_rekey(), 0);

        // Extract responder's pubkey and nonce from the ack
        let (resp_pubkey, resp_nonce, ack_seq) = match &ack_payload {
            TunnelPayload::RekeyAck { ephemeral_pubkey, nonce, rekey_seq } => {
                (*ephemeral_pubkey, *nonce, *rekey_seq)
            }
            _ => panic!("Expected TunnelPayload::RekeyAck"),
        };
        assert_eq!(ack_seq, rekey_seq);

        // Step 3: Initiator completes the rekey
        initiator.complete_rekey(&resp_pubkey, &resp_nonce, ack_seq).unwrap();
        assert_eq!(initiator.state(), RekeyState::Idle);
        assert_eq!(initiator.rekey_count(), 1);
        assert_eq!(initiator.bytes_since_rekey(), 0);

        // Both sides should now have the same NEW session key
        assert_eq!(
            initiator.current_key().unwrap().as_bytes(),
            responder.current_key().unwrap().as_bytes(),
        );

        // The new key should differ from the initial key
        assert_ne!(
            initiator.current_key().unwrap().as_bytes(),
            &[0x10u8; 32],
        );
    }

    #[test]
    fn test_rekey_cannot_initiate_while_awaiting_ack() {
        let key = make_test_session_key(0x20);
        let mut mgr = RekeyManager::new(key);

        // First initiation succeeds
        let _ = mgr.initiate_rekey().unwrap();
        assert_eq!(mgr.state(), RekeyState::AwaitingAck);

        // Second initiation fails
        let result = mgr.initiate_rekey();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("AwaitingAck"));
    }

    #[test]
    fn test_rekey_failure_preserves_old_key() {
        let key = make_test_session_key(0x30);
        let mut mgr = RekeyManager::new(key);

        // Initiate and then fail
        let _ = mgr.initiate_rekey().unwrap();
        mgr.fail_rekey("test failure");

        assert_eq!(mgr.state(), RekeyState::Failed);
        // Old key should still be there
        assert!(mgr.current_key().is_some());
        assert_eq!(mgr.current_key().unwrap().as_bytes(), &[0x30; 32]);

        // Should not trigger rekey immediately (retry interval not elapsed)
        assert!(!mgr.should_rekey());
    }

    #[test]
    fn test_rekey_seq_mismatch_rejected() {
        let key = make_test_session_key(0x40);
        let mut mgr = RekeyManager::new(key);

        let _ = mgr.initiate_rekey().unwrap();

        // Try to complete with wrong seq
        let result = mgr.complete_rekey(&[0u8; 32], &[0u8; 32], 999);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("seq mismatch"));
    }

    #[test]
    fn test_rekey_stats_snapshot() {
        let key = make_test_session_key(0x50);
        let mut mgr = RekeyManager::new(key);
        mgr.record_bytes(12345);

        let stats = mgr.stats(100_000, 200_000, 50, 15);
        assert_eq!(stats.bytes_sent, 100_000);
        assert_eq!(stats.bytes_received, 200_000);
        assert_eq!(stats.tasks_routed, 50);
        assert_eq!(stats.latency_ms, 15);
        assert_eq!(stats.rekey_count, 0);
        assert_eq!(stats.last_rekey_time, 0); // no rekey yet
        assert_eq!(stats.bytes_since_rekey, 12345);
    }

    #[test]
    fn test_rekey_multiple_sequential_rekeys() {
        // Perform 3 sequential rekeys and verify counters and key changes.
        let initial_key = make_test_session_key(0x60);
        let mut initiator = RekeyManager::new(initial_key.clone());
        let mut responder = RekeyManager::new(initial_key);

        let mut prev_key_bytes = *initiator.current_key().unwrap().as_bytes();

        for expected_count in 1..=3u64 {
            // Initiator starts rekey
            let rekey_payload = initiator.initiate_rekey().unwrap();
            let (pk, nonce, seq) = match &rekey_payload {
                TunnelPayload::Rekey { ephemeral_pubkey, nonce, rekey_seq } => {
                    (*ephemeral_pubkey, *nonce, *rekey_seq)
                }
                _ => panic!("Expected Rekey"),
            };

            // Responder handles
            let ack = responder.handle_rekey_request(&pk, &nonce, seq).unwrap();
            let (rpk, rn, rs) = match &ack {
                TunnelPayload::RekeyAck { ephemeral_pubkey, nonce, rekey_seq } => {
                    (*ephemeral_pubkey, *nonce, *rekey_seq)
                }
                _ => panic!("Expected RekeyAck"),
            };

            // Initiator completes
            initiator.complete_rekey(&rpk, &rn, rs).unwrap();

            assert_eq!(initiator.rekey_count(), expected_count);
            assert_eq!(responder.rekey_count(), expected_count);

            // Keys match on both sides
            assert_eq!(
                initiator.current_key().unwrap().as_bytes(),
                responder.current_key().unwrap().as_bytes(),
            );

            // Key changed from previous round
            let new_key_bytes = *initiator.current_key().unwrap().as_bytes();
            assert_ne!(new_key_bytes, prev_key_bytes);
            prev_key_bytes = new_key_bytes;
        }
    }

    #[test]
    fn test_session_key_zeroize_on_drop() {
        // Verify that SessionKey implements Zeroize by creating one and
        // checking that zeroize() clears the key bytes.
        let mut key = make_test_session_key(0xFF);
        assert_eq!(key.as_bytes(), &[0xFF; 32]);

        // Manually zeroize
        key.zeroize();
        assert_eq!(key.as_bytes(), &[0u8; 32]);
    }

    #[test]
    fn test_rekey_reset_failure_enables_retry() {
        let key = make_test_session_key(0x70);
        let mut mgr = RekeyManager::new(key);

        // Initiate and fail
        let _ = mgr.initiate_rekey().unwrap();
        mgr.fail_rekey("test");
        assert_eq!(mgr.state(), RekeyState::Failed);

        // Should not rekey while failed (retry interval not elapsed)
        assert!(!mgr.should_rekey());

        // Reset failure state
        mgr.reset_failure();
        assert_eq!(mgr.state(), RekeyState::Idle);
    }

    #[tokio::test]
    async fn test_tunnel_stream_rekey_sends_message() {
        let (tx, mut rx) = mpsc::channel(16);
        let stream = TunnelStream {
            stream_type: StreamType::Control,
            stream_id: 1,
            tx,
            created_at: Instant::now(),
            bytes_sent: Arc::new(AtomicU64::new(0)),
            bytes_received: Arc::new(AtomicU64::new(0)),
        };

        let initial_key = make_test_session_key(0x80);
        let mut mgr = RekeyManager::new(initial_key);

        // Perform rekey via the stream
        let payload = stream.rekey(&mut mgr).await.unwrap();
        assert_eq!(mgr.state(), RekeyState::AwaitingAck);

        // Verify the message was sent through the channel
        let received = rx.recv().await.unwrap();
        match received {
            TunnelPayload::Rekey { ephemeral_pubkey, nonce, rekey_seq } => {
                assert_eq!(rekey_seq, 1);
                assert_ne!(ephemeral_pubkey, [0u8; 32]); // Should have real key
                assert_ne!(nonce, [0u8; 32]); // Should have real nonce
            }
            _ => panic!("Expected TunnelPayload::Rekey, got {:?}", received),
        }

        // Verify bytes_sent was updated
        assert_eq!(stream.total_bytes_sent(), 72);
    }

    #[test]
    fn test_derive_rekey_session_key_deterministic() {
        // Same inputs should produce the same key
        let shared_secret = [0x42u8; 32];
        let nonce_a = [0x01u8; 32];
        let nonce_b = [0x02u8; 32];
        let seq = 5u64;

        let key1 = derive_rekey_session_key(&shared_secret, &nonce_a, &nonce_b, seq);
        let key2 = derive_rekey_session_key(&shared_secret, &nonce_a, &nonce_b, seq);
        assert_eq!(key1.as_bytes(), key2.as_bytes());

        // Different seq should produce different key
        let key3 = derive_rekey_session_key(&shared_secret, &nonce_a, &nonce_b, seq + 1);
        assert_ne!(key1.as_bytes(), key3.as_bytes());

        // Swapped nonces should produce different key
        let key4 = derive_rekey_session_key(&shared_secret, &nonce_b, &nonce_a, seq);
        assert_ne!(key1.as_bytes(), key4.as_bytes());
    }

    #[test]
    fn test_rekey_domain_separation_from_initial_handshake() {
        // Verify that derive_rekey_session_key produces different keys than
        // derive_session_key even with the same inputs (domain separation
        // via different HKDF info strings).
        let shared_secret = [0x99u8; 32];
        let nonce_a = [0x11u8; 32];
        let nonce_b = [0x22u8; 32];

        let initial_key = derive_session_key(&shared_secret, &nonce_a, &nonce_b);
        let rekey_key = derive_rekey_session_key(&shared_secret, &nonce_a, &nonce_b, 1);

        assert_ne!(initial_key.as_bytes(), rekey_key.as_bytes());
    }

    #[test]
    fn test_tunnel_payload_rekey_variant_serialization() {
        let payload = TunnelPayload::Rekey {
            ephemeral_pubkey: [0xAB; 32],
            nonce: [0xCD; 32],
            rekey_seq: 42,
        };

        let json = serde_json::to_string(&payload).unwrap();
        let deserialized: TunnelPayload = serde_json::from_str(&json).unwrap();

        match deserialized {
            TunnelPayload::Rekey { ephemeral_pubkey, nonce, rekey_seq } => {
                assert_eq!(ephemeral_pubkey, [0xAB; 32]);
                assert_eq!(nonce, [0xCD; 32]);
                assert_eq!(rekey_seq, 42);
            }
            _ => panic!("Deserialization produced wrong variant"),
        }
    }

    #[test]
    fn test_tunnel_payload_rekey_ack_serialization() {
        let payload = TunnelPayload::RekeyAck {
            ephemeral_pubkey: [0xEF; 32],
            nonce: [0x12; 32],
            rekey_seq: 7,
        };

        let json = serde_json::to_string(&payload).unwrap();
        let deserialized: TunnelPayload = serde_json::from_str(&json).unwrap();

        match deserialized {
            TunnelPayload::RekeyAck { ephemeral_pubkey, nonce, rekey_seq } => {
                assert_eq!(ephemeral_pubkey, [0xEF; 32]);
                assert_eq!(nonce, [0x12; 32]);
                assert_eq!(rekey_seq, 7);
            }
            _ => panic!("Deserialization produced wrong variant"),
        }
    }
}
