//! WebRTC Signaling Server
//!
//! Routes SDP offers/answers and ICE candidates between browser peers
//! for voice, video, and meeting sessions.
//!
//! Protocol (nova-chat compatible):
//!   - Peers connect to /ws/chat/signal?peer_id=<wallet_address>
//!   - Messages are JSON-encoded SignalingEnvelope
//!   - Server routes envelopes to the target peer's WebSocket
//!   - Meeting rooms: server broadcasts to all room members

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::call_manager::{CallManager, CALL_INITIATION_TIMEOUT_SECS};
use crate::wallet_auth::validate_signaling_auth_query;

// ─── Limits ──────────────────────────────────────────────────────────────────

/// Maximum peer_id length — rejects oversized query params before state insertion.
const MAX_PEER_ID_LEN: usize = 256;
/// Maximum peers allowed per meeting room.
const MAX_ROOM_CAPACITY: usize = 49;
/// Maximum meeting rooms a single peer may join simultaneously.
const MAX_ROOMS_PER_PEER: usize = 5;
/// Outbound channel capacity per peer — slow receivers are disconnected, not OOM'd.
const PEER_CHANNEL_CAPACITY: usize = 256;

// ─── Types ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SignalingPayload {
    /// WebRTC SDP offer (caller → callee)
    CallOffer { sdp: String, call_type: CallType },
    /// WebRTC SDP answer (callee → caller)
    CallAnswer { sdp: String },
    /// ICE candidate exchange
    IceCandidate { candidate: String, sdp_mid: Option<String>, sdp_m_line_index: Option<u16> },
    /// Call ended by either party
    CallEnd { reason: Option<String> },
    /// Chat message (text)
    ChatMessage { content: String, timestamp: u64 },
    /// Meeting room: join
    MeetingJoin { room_id: String, display_name: String },
    /// Meeting room: leave
    MeetingLeave { room_id: String },
    /// Server → client: list of peers in a meeting room
    MeetingPeers { room_id: String, peers: Vec<PeerInfo> },
    /// Server → client: a peer joined the room
    MeetingPeerJoined { room_id: String, peer: PeerInfo },
    /// Server → client: a peer left the room
    MeetingPeerLeft { room_id: String, peer_id: String },
    /// Ping keepalive
    Ping,
    /// Pong response
    Pong,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallType {
    Audio,
    Video,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub peer_id: String,
    pub display_name: String,
}

/// Every message on the wire is wrapped in this envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalingEnvelope {
    /// Sender's wallet address / peer ID
    pub from: String,
    /// Target peer ID (None = broadcast to room)
    pub to: Option<String>,
    /// Optional call/meeting ID for correlation
    pub session_id: Option<String>,
    pub payload: SignalingPayload,
}

// ─── State ───────────────────────────────────────────────────────────────────

/// Bounded channel — slow receivers are dropped rather than allowed to OOM the server (CRIT-2 fix).
type PeerSender = mpsc::Sender<SignalingEnvelope>;

#[derive(Clone)]
pub struct SignalingState {
    /// peer_id → message sender channel
    peers: Arc<DashMap<String, PeerSender>>,
    /// room_id → set of peer_ids
    rooms: Arc<DashMap<String, Vec<PeerInfo>>>,
    /// peer_id → room_ids the peer has joined (for O(1) cleanup and per-peer room limit)
    peer_rooms: Arc<DashMap<String, Vec<String>>>,
    /// Call lifecycle: capacity enforcement, per-peer dedup, timeout detection
    call_manager: Arc<CallManager>,
}

impl SignalingState {
    pub fn new() -> Self {
        Self {
            peers: Arc::new(DashMap::new()),
            rooms: Arc::new(DashMap::new()),
            peer_rooms: Arc::new(DashMap::new()),
            call_manager: Arc::new(CallManager::new()),
        }
    }
}

impl Default for SignalingState {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Query params ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct SignalQuery {
    pub peer_id: String,
    /// URL-encoded X-Wallet-Auth JSON — Ed25519 signature over `/ws/chat/signal` (CRIT-1 fix)
    pub auth_header: Option<String>,
    pub display_name: Option<String>,
}

// ─── Handler ──────────────────────────────────────────────────────────────────

pub async fn ws_signal_handler(
    ws: WebSocketUpgrade,
    Query(query): Query<SignalQuery>,
    State(state): State<SignalingState>,
) -> Response {
    let short = short_peer(&query.peer_id);

    // ── CRIT-1: Verify ownership of peer_id via Ed25519 signature ─────────────
    if let Some(auth_json) = &query.auth_header {
        match validate_signaling_auth_query(auth_json, &query.peer_id) {
            Ok(_) => { /* authenticated */ }
            Err(reason) => {
                warn!(
                    "Signaling: auth REJECTED for peer_id={} reason={} auth_len={}",
                    short,
                    reason,
                    auth_json.len()
                );
                return (StatusCode::UNAUTHORIZED, reason).into_response();
            }
        }
    } else {
        warn!("Signaling: connect REJECTED for peer_id={} reason=missing_auth_header", short);
        return (StatusCode::UNAUTHORIZED, "auth_header query parameter required").into_response();
    }

    // ── CRIT-2: Sanitise peer_id before accepting the connection ─────────────
    let peer_id = query.peer_id.trim().to_string();
    if peer_id.is_empty() || peer_id.len() > MAX_PEER_ID_LEN {
        warn!(
            "Signaling: connect REJECTED for peer_id={} reason=invalid_length (len={})",
            short,
            peer_id.len()
        );
        return (StatusCode::BAD_REQUEST, "peer_id invalid or too long").into_response();
    }

    // Reject duplicate peer_id — prevents session hijacking (CRIT-1 supplement)
    if state.peers.contains_key(&peer_id) {
        warn!(
            "Signaling: duplicate peer_id rejected — {} (already connected; current peer_count={})",
            short,
            state.peers.len()
        );
        return (StatusCode::CONFLICT, "peer_id already connected").into_response();
    }

    info!(
        "Signaling: connect ACCEPTED peer_id={} (peer_count_before={})",
        short,
        state.peers.len()
    );
    ws.on_upgrade(move |socket| handle_peer(socket, peer_id, state))
}

/// Short representation of a peer_id for logs — first 8 and last 6 chars.
fn short_peer(peer_id: &str) -> String {
    if peer_id.len() <= 16 {
        return peer_id.to_string();
    }
    format!("{}…{}", &peer_id[..8], &peer_id[peer_id.len() - 6..])
}

// ─── Diagnostic endpoint ─────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct SignalingDiagResponse {
    pub peer_count: usize,
    pub peers: Vec<String>,
    pub room_count: usize,
    pub active_call_count: usize,
}

/// GET /api/v1/signaling/diag — lists currently-connected signaling peers.
///
/// Public diagnostic endpoint: lets callers verify whether the peer they're
/// trying to reach is actually connected to *this* backend (vs. a different
/// load-balanced one). Returns truncated peer_ids — full IDs are not needed
/// for diagnostics and shouldn't leak via this endpoint.
pub async fn signaling_diag_handler(
    State(state): State<SignalingState>,
) -> Json<SignalingDiagResponse> {
    let peers: Vec<String> = state
        .peers
        .iter()
        .map(|p| short_peer(p.key()))
        .collect();
    Json(SignalingDiagResponse {
        peer_count: state.peers.len(),
        peers,
        room_count: state.rooms.len(),
        active_call_count: state.call_manager.active_call_count(),
    })
}

async fn handle_peer(socket: WebSocket, peer_id: String, state: SignalingState) {
    info!("🔌 Signaling: peer connected — {}", peer_id);

    let (mut ws_tx, mut ws_rx) = socket.split();
    // Bounded channel — prevents OOM when a slow/frozen client buffers unboundedly (CRIT-2 fix)
    let (msg_tx, mut msg_rx) = mpsc::channel::<SignalingEnvelope>(PEER_CHANNEL_CAPACITY);

    // Register this peer
    state.peers.insert(peer_id.clone(), msg_tx);

    // Task: forward outbound messages to the WebSocket
    let peer_id_clone = peer_id.clone();
    let forward_task = tokio::spawn(async move {
        while let Some(envelope) = msg_rx.recv().await {
            match serde_json::to_string(&envelope) {
                Ok(json) => {
                    if ws_tx.send(Message::Text(json.into())).await.is_err() {
                        break;
                    }
                }
                Err(e) => warn!("Signaling: serialization error for {}: {}", peer_id_clone, e),
            }
        }
    });
    // Note: msg_rx closes when all senders are dropped. If the bounded channel fills up,
    // send attempts return Err and callers log a warning + skip delivery (no OOM).

    // Task: detect calls stuck in Initiating state and send CallEnd to both peers
    let timeout_state = state.clone();
    let timeout_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
        loop {
            interval.tick().await;
            let timed_out = timeout_state.call_manager.get_timed_out_calls(CALL_INITIATION_TIMEOUT_SECS);
            for (call_id, caller, callee) in timed_out {
                warn!("Signaling: call {} timed out (Initiating > {}s)", call_id, CALL_INITIATION_TIMEOUT_SECS);
                timeout_state.call_manager.end_call(&call_id);
                let end = SignalingEnvelope {
                    from: "server".to_string(),
                    to: None,
                    session_id: Some(call_id.clone()),
                    payload: SignalingPayload::CallEnd { reason: Some("timeout".to_string()) },
                };
                for pid in [&caller, &callee] {
                    if pid.is_empty() { continue; }
                    if let Some(sender) = timeout_state.peers.get(pid.as_str()) {
                        let mut env = end.clone();
                        env.to = Some((*pid).to_string());
                        let _ = sender.try_send(env);
                    }
                }
            }
        }
    });

    // Main loop: receive from WebSocket, route to target peer
    while let Some(Ok(msg)) = ws_rx.next().await {
        match msg {
            Message::Text(text) => {
                match serde_json::from_str::<SignalingEnvelope>(&text) {
                    Ok(mut envelope) => {
                        envelope.from = peer_id.clone();
                        route_message(&state, envelope).await;
                    }
                    Err(e) => warn!("Signaling: parse error from {}: {}", peer_id, e),
                }
            }
            Message::Close(_) => break,
            Message::Ping(data) => {
                // axum handles pong automatically
                let _ = data;
            }
            _ => {}
        }
    }

    // Abort tasks first, then clean up state (MED-5: prevents post-cleanup sends to dead socket)
    forward_task.abort();
    timeout_task.abort();
    cleanup_peer(&state, &peer_id).await;
    info!("🔌 Signaling: peer disconnected — {}", peer_id);
}

async fn route_message(state: &SignalingState, envelope: SignalingEnvelope) {
    match &envelope.payload {
        SignalingPayload::Ping => {
            // Send pong back to sender
            if let Some(sender) = state.peers.get(&envelope.from) {
                let pong = SignalingEnvelope {
                    from: "server".to_string(),
                    to: Some(envelope.from.clone()),
                    session_id: envelope.session_id.clone(),
                    payload: SignalingPayload::Pong,
                };
                let _ = sender.try_send(pong);
            }
            return;
        }

        SignalingPayload::MeetingJoin { room_id, display_name } => {
            let room_id = room_id.clone();
            let display_name = display_name.clone();
            let peer_info = PeerInfo {
                peer_id: envelope.from.clone(),
                display_name: display_name.clone(),
            };

            // ── CRIT-2: Per-peer room limit ───────────────────────────────────
            let room_count = state.peer_rooms
                .get(&envelope.from)
                .map(|r| r.len())
                .unwrap_or(0);
            if room_count >= MAX_ROOMS_PER_PEER {
                warn!("Signaling: {} exceeded max rooms per peer ({})", envelope.from, MAX_ROOMS_PER_PEER);
                return;
            }

            // ── CRIT-2: Room capacity enforcement ────────────────────────────
            let already_in_room = state.rooms
                .get(&room_id)
                .map(|r| r.iter().any(|p| p.peer_id == envelope.from))
                .unwrap_or(false);
            let current_capacity = state.rooms.get(&room_id).map(|r| r.len()).unwrap_or(0);
            if !already_in_room && current_capacity >= MAX_ROOM_CAPACITY {
                warn!("Signaling: room {} at capacity ({})", room_id, MAX_ROOM_CAPACITY);
                return;
            }

            // Track which rooms this peer is in (for O(1) cleanup on disconnect)
            state.peer_rooms
                .entry(envelope.from.clone())
                .or_default()
                .retain(|r| r != &room_id); // idempotent re-join
            state.peer_rooms
                .entry(envelope.from.clone())
                .or_default()
                .push(room_id.clone());

            // Add to room
            let existing_peers = {
                let mut room = state.rooms.entry(room_id.clone()).or_default();
                room.retain(|p| p.peer_id != envelope.from); // remove stale entry
                room.push(peer_info.clone());
                room.clone()
            };

            // Tell the joiner about existing peers
            if let Some(sender) = state.peers.get(&envelope.from) {
                let _ = sender.try_send(SignalingEnvelope {
                    from: "server".to_string(),
                    to: Some(envelope.from.clone()),
                    session_id: None,
                    payload: SignalingPayload::MeetingPeers {
                        room_id: room_id.clone(),
                        peers: existing_peers.iter()
                            .filter(|p| p.peer_id != envelope.from)
                            .cloned()
                            .collect(),
                    },
                });
            }

            // Notify existing peers of the newcomer
            let peers_snapshot: Vec<String> = state.rooms
                .get(&room_id)
                .map(|r| r.iter().map(|p| p.peer_id.clone()).collect())
                .unwrap_or_default();

            for pid in peers_snapshot {
                if pid != envelope.from {
                    if let Some(sender) = state.peers.get(&pid) {
                        let _ = sender.try_send(SignalingEnvelope {
                            from: "server".to_string(),
                            to: Some(pid.clone()),
                            session_id: None,
                            payload: SignalingPayload::MeetingPeerJoined {
                                room_id: room_id.clone(),
                                peer: peer_info.clone(),
                            },
                        });
                    }
                }
            }
            return;
        }

        SignalingPayload::MeetingLeave { room_id } => {
            remove_from_room(state, room_id, &envelope.from).await;
            return;
        }

        _ => {}
    }

    // Call lifecycle management
    match &envelope.payload {
        SignalingPayload::CallOffer { call_type, .. } => {
            if let Some(callee) = &envelope.to {
                let call_id = envelope.session_id.clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                let call_type_str = match call_type {
                    CallType::Audio => "audio",
                    CallType::Video => "video",
                };
                match state.call_manager.register_call(&call_id, &envelope.from, callee, call_type_str) {
                    Err("capacity") => {
                        warn!("Signaling: call rejected — server at capacity ({})", envelope.from);
                        if let Some(sender) = state.peers.get(&envelope.from) {
                            let _ = sender.try_send(SignalingEnvelope {
                                from: "server".to_string(),
                                to: Some(envelope.from.clone()),
                                session_id: envelope.session_id.clone(),
                                payload: SignalingPayload::CallEnd { reason: Some("server_capacity".to_string()) },
                            });
                        }
                        return;
                    }
                    Err("busy") => {
                        warn!("Signaling: call rejected — peer busy ({} → {})", envelope.from, callee);
                        if let Some(sender) = state.peers.get(&envelope.from) {
                            let _ = sender.try_send(SignalingEnvelope {
                                from: "server".to_string(),
                                to: Some(envelope.from.clone()),
                                session_id: envelope.session_id.clone(),
                                payload: SignalingPayload::CallEnd { reason: Some("peer_busy".to_string()) },
                            });
                        }
                        return;
                    }
                    _ => {}
                }
            }
        }
        SignalingPayload::CallAnswer { .. } => {
            if let Some(call_id) = &envelope.session_id {
                state.call_manager.mark_connected(call_id);
            }
        }
        SignalingPayload::CallEnd { .. } => {
            if let Some(call_id) = &envelope.session_id {
                state.call_manager.end_call(call_id);
            } else if let Some(call_id) = state.call_manager.find_call_for_peer(&envelope.from) {
                state.call_manager.end_call(&call_id);
            }
        }
        _ => {}
    }

    // Direct peer routing
    if let Some(target) = &envelope.to {
        let msg_type = match &envelope.payload {
            SignalingPayload::CallOffer { .. } => "call_offer",
            SignalingPayload::CallAnswer { .. } => "call_answer",
            SignalingPayload::CallEnd { .. } => "call_end",
            SignalingPayload::IceCandidate { .. } => "ice_candidate",
            SignalingPayload::ChatMessage { .. } => "chat_message",
            _ => "other",
        };
        let from_short = short_peer(&envelope.from);
        let to_short = short_peer(target);
        if let Some(sender) = state.peers.get(target) {
            match sender.try_send(envelope.clone()) {
                Ok(_) => {
                    // Call-control envelopes are rare and useful to see; chat / ICE are noisy.
                    if matches!(
                        msg_type,
                        "call_offer" | "call_answer" | "call_end"
                    ) {
                        info!(
                            "Signaling: routed {} {} → {} (sid={:?})",
                            msg_type, from_short, to_short, envelope.session_id
                        );
                    } else {
                        debug!(
                            "Signaling: routed {} {} → {} (sid={:?})",
                            msg_type, from_short, to_short, envelope.session_id
                        );
                    }
                }
                Err(e) => warn!(
                    "Signaling: DROP {} {} → {} (channel full or closed — {})",
                    msg_type, from_short, to_short, e
                ),
            }
        } else {
            // Compact preview of who IS connected so debugging "where's my peer?" is one log away.
            let peer_count = state.peers.len();
            let preview: Vec<String> = state
                .peers
                .iter()
                .take(8)
                .map(|p| short_peer(p.key()))
                .collect();
            warn!(
                "Signaling: target NOT CONNECTED {} {} → {} (peer_count={} connected_preview={:?})",
                msg_type, from_short, to_short, peer_count, preview
            );
            // Notify sender that target is offline
            if let Some(from_sender) = state.peers.get(&envelope.from) {
                let _ = from_sender.try_send(SignalingEnvelope {
                    from: "server".to_string(),
                    to: Some(envelope.from.clone()),
                    session_id: envelope.session_id.clone(),
                    payload: SignalingPayload::CallEnd { reason: Some("peer_not_found".to_string()) },
                });
            }
        }
    }
}

async fn remove_from_room(state: &SignalingState, room_id: &str, peer_id: &str) {
    // Update peer→room index (O(1) cleanup path)
    state.peer_rooms.alter(peer_id, |_, mut rooms| {
        rooms.retain(|r| r != room_id);
        rooms
    });
    state.peer_rooms.retain(|_, rooms| !rooms.is_empty());

    state.rooms.alter(room_id, |_, mut peers| {
        peers.retain(|p| p.peer_id != peer_id);
        peers
    });

    // Notify remaining peers
    let remaining: Vec<String> = state.rooms
        .get(room_id)
        .map(|r| r.iter().map(|p| p.peer_id.clone()).collect())
        .unwrap_or_default();

    for pid in remaining {
        if let Some(sender) = state.peers.get(&pid) {
            let _ = sender.try_send(SignalingEnvelope {
                from: "server".to_string(),
                to: Some(pid.clone()),
                session_id: None,
                payload: SignalingPayload::MeetingPeerLeft {
                    room_id: room_id.to_string(),
                    peer_id: peer_id.to_string(),
                },
            });
        }
    }

    // Clean up empty rooms
    state.rooms.retain(|_, v| !v.is_empty());
}

async fn cleanup_peer(state: &SignalingState, peer_id: &str) {
    // Atomically claim cleanup — only the first caller proceeds (HIGH-2: concurrent disconnect safety)
    if state.peers.remove(peer_id).is_none() {
        return;
    }

    // End any active call this peer was in, notifying the other party
    if let Some(call_id) = state.call_manager.find_call_for_peer(peer_id) {
        if let Some((caller, callee)) = state.call_manager.get_call_peers(&call_id) {
            let other = if caller == peer_id { callee } else { caller };
            if !other.is_empty() {
                if let Some(sender) = state.peers.get(&other) {
                    let _ = sender.try_send(SignalingEnvelope {
                        from: "server".to_string(),
                        to: Some(other.clone()),
                        session_id: Some(call_id.clone()),
                        payload: SignalingPayload::CallEnd { reason: Some("peer_disconnected".to_string()) },
                    });
                }
            }
        }
        state.call_manager.end_call(&call_id);
    }

    // Remove from all rooms using O(1) peer→room index (LOW-5 fix — was O(R×P))
    let room_ids: Vec<String> = state.peer_rooms
        .remove(peer_id)
        .map(|(_, rooms)| rooms)
        .unwrap_or_default();

    for room_id in room_ids {
        remove_from_room(state, &room_id, peer_id).await;
    }
}
