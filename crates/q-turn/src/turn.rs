// TURN protocol handler (RFC 5766).
//
// Entry point: `handle_message(raw, src, state)`.
// Returns `Some(response_bytes)` to send back to `src`, or `None`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UdpSocket;
use tracing::{debug, info, warn};

use crate::allocation::{Allocation, AllocationStore, PortPool};
use crate::auth::AuthState;
use crate::config::Config;
use crate::stun::{
    self,
    StunMessage,
    ATTR_CHANNEL_NUMBER, ATTR_DATA, ATTR_LIFETIME,
    ATTR_NONCE, ATTR_REALM, ATTR_REQUESTED_TRANSPORT, ATTR_USERNAME,
    ATTR_XOR_MAPPED_ADDRESS, ATTR_XOR_PEER_ADDRESS, ATTR_XOR_RELAYED_ADDRESS,
    ERR_BAD_REQUEST, ERR_QUOTA_REACHED,
    ERR_UNAUTHORIZED, ERR_UNSUPPORTED_PROTO, MSG_BINDING_REQUEST, MSG_BINDING_SUCCESS,
    MSG_ALLOCATE_REQUEST, MSG_ALLOCATE_SUCCESS, MSG_ALLOCATE_ERROR,
    MSG_CHANNEL_BIND_REQUEST, MSG_CHANNEL_BIND_SUCCESS,
    MSG_CREATE_PERM_REQUEST, MSG_CREATE_PERM_SUCCESS,
    MSG_DATA_INDICATION, MSG_REFRESH_REQUEST, MSG_REFRESH_SUCCESS, MSG_REFRESH_ERROR,
    MSG_SEND_INDICATION, TRANSPORT_UDP, decode_xor_addr, encode_channel_data,
};

pub struct TurnState {
    pub auth:    Arc<AuthState>,
    pub allocs:  Arc<AllocationStore>,
    pub ports:   Arc<PortPool>,
    pub config:  Config,
    pub main_sock: Arc<UdpSocket>,
}

impl TurnState {
    pub fn new(config: Config, main_sock: Arc<UdpSocket>) -> Arc<Self> {
        let auth   = AuthState::new(
            config.auth.secret.clone(),
            config.server.realm.clone(),
            config.auth.credential_ttl,
        );
        let allocs = AllocationStore::new(config.limits.max_allocations);
        let ports  = PortPool::new(config.relay.min_port, config.relay.max_port);
        Arc::new(Self { auth, allocs, ports, config, main_sock })
    }
}

/// Process an incoming UDP packet from `src`.
/// Returns bytes to send back to `src`, or `None`.
pub async fn handle_packet(
    raw: &[u8],
    src: SocketAddr,
    state: &Arc<TurnState>,
) -> Option<Vec<u8>> {
    // Distinguish STUN message vs ChannelData
    if stun::is_channel_data(raw) {
        return handle_channel_data(raw, src, state).await;
    }

    let (msg, _) = StunMessage::parse(raw)?;

    match msg.msg_type {
        MSG_BINDING_REQUEST      => Some(handle_binding(&msg, src, state)),
        MSG_ALLOCATE_REQUEST     => handle_allocate(raw, msg, src, state).await,
        MSG_REFRESH_REQUEST      => handle_refresh(raw, msg, src, state),
        MSG_CREATE_PERM_REQUEST  => handle_create_permission(raw, msg, src, state),
        MSG_CHANNEL_BIND_REQUEST => handle_channel_bind(raw, msg, src, state),
        MSG_SEND_INDICATION      => { handle_send_indication(msg, src, state).await; None }
        _                        => None,
    }
}

// ─── Binding (STUN NAT check) ─────────────────────────────────────────────────

fn handle_binding(msg: &StunMessage, src: SocketAddr, state: &TurnState) -> Vec<u8> {
    let mut resp = StunMessage::new(MSG_BINDING_SUCCESS, msg.transaction_id);
    resp.add_xor_addr(ATTR_XOR_MAPPED_ADDRESS, src);
    resp.encode_simple(&state.config.server.software)
}

// ─── Auth helpers ─────────────────────────────────────────────────────────────

/// Extract credentials and derive MI key.  Returns `Err(response_bytes)` on failure.
fn check_auth(
    raw: &[u8],
    msg: &StunMessage,
    src: SocketAddr,
    state: &TurnState,
    error_type: u16,
) -> Result<(String, Vec<u8>), Vec<u8>> {
    let username = msg.get_attr(ATTR_USERNAME)
        .and_then(|b| std::str::from_utf8(b).ok())
        .map(|s| s.to_string());
    let realm = msg.get_attr(ATTR_REALM)
        .and_then(|b| std::str::from_utf8(b).ok())
        .map(|s| s.to_string());
    let nonce = msg.get_attr(ATTR_NONCE)
        .and_then(|b| std::str::from_utf8(b).ok())
        .map(|s| s.to_string());

    // Missing credentials → 401 with fresh nonce
    let (Some(username), Some(realm), Some(nonce)) = (username, realm, nonce) else {
        return Err(make_auth_challenge(msg, state, error_type));
    };

    // Validate credentials
    let Some(mi_key) = state.auth.validate_credentials(&username, &realm, &nonce) else {
        return Err(make_auth_challenge(msg, state, error_type));
    };

    // Verify MESSAGE-INTEGRITY
    let (Some(mi_offset), Some(mi_value)) = (msg.mi_offset, msg.mi_value) else {
        return Err(make_auth_challenge(msg, state, error_type));
    };
    if !stun::verify_message_integrity(raw, mi_offset, &mi_value, &mi_key) {
        let mut err = StunMessage::new(error_type, msg.transaction_id);
        err.add_error(ERR_UNAUTHORIZED.0, ERR_UNAUTHORIZED.1);
        return Err(err.encode_simple(&state.config.server.software));
    }

    Ok((username, mi_key))
}

fn make_auth_challenge(msg: &StunMessage, state: &TurnState, error_type: u16) -> Vec<u8> {
    let nonce = state.auth.generate_nonce();
    let mut err = StunMessage::new(error_type, msg.transaction_id);
    err.add_error(ERR_UNAUTHORIZED.0, ERR_UNAUTHORIZED.1);
    err.add_str(ATTR_REALM, state.auth.realm());
    err.add_str(ATTR_NONCE, &nonce);
    err.encode_simple(&state.config.server.software)
}

// ─── Allocate ─────────────────────────────────────────────────────────────────

async fn handle_allocate(
    raw: &[u8],
    msg: StunMessage,
    src: SocketAddr,
    state: &Arc<TurnState>,
) -> Option<Vec<u8>> {
    // Check for existing allocation — send error if already exists
    if state.allocs.has(src) {
        let mut err = StunMessage::new(MSG_ALLOCATE_ERROR, msg.transaction_id);
        err.add_error(437, "Allocation Mismatch");
        return Some(err.encode_simple(&state.config.server.software));
    }

    let (username, mi_key) = match check_auth(raw, &msg, src, state, MSG_ALLOCATE_ERROR) {
        Ok(v)  => v,
        Err(e) => return Some(e),
    };

    // Validate REQUESTED-TRANSPORT (must be UDP = 17)
    let transport = msg.get_attr(ATTR_REQUESTED_TRANSPORT)
        .and_then(|b| b.first().copied());
    if transport != Some(TRANSPORT_UDP) {
        let mut err = StunMessage::new(MSG_ALLOCATE_ERROR, msg.transaction_id);
        err.add_error(ERR_UNSUPPORTED_PROTO.0, ERR_UNSUPPORTED_PROTO.1);
        return Some(err.encode_with_integrity(Some(&mi_key), &state.config.server.software));
    }

    // Requested lifetime (capped to server max)
    let requested_lifetime = msg.get_attr(ATTR_LIFETIME)
        .filter(|b| b.len() == 4)
        .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as u64)
        .unwrap_or(state.config.limits.allocation_lifetime);
    let lifetime = requested_lifetime.min(state.config.limits.allocation_lifetime);

    // Allocate a relay socket
    let Some((relay_sock, relay_addr)) = state.ports
        .allocate_relay_socket(state.config.relay.public_ip).await else {
        let mut err = StunMessage::new(MSG_ALLOCATE_ERROR, msg.transaction_id);
        err.add_error(ERR_INSUF_CAPACITY.0, ERR_INSUF_CAPACITY.1);
        return Some(err.encode_with_integrity(Some(&mi_key), &state.config.server.software));
    };

    let alloc = Allocation::new(
        relay_sock.clone(),
        relay_addr,
        src,
        username,
        Duration::from_secs(lifetime),
    );

    if !state.allocs.insert(alloc.clone()) {
        let mut err = StunMessage::new(MSG_ALLOCATE_ERROR, msg.transaction_id);
        err.add_error(ERR_QUOTA_REACHED.0, ERR_QUOTA_REACHED.1);
        return Some(err.encode_with_integrity(Some(&mi_key), &state.config.server.software));
    }

    info!("TURN alloc: {} → relay:{} ({}s)", src, relay_addr, lifetime);

    // Spawn relay task: forwards data from peers to this client
    let state_arc = state.clone();
    let alloc_for_task = alloc.clone();
    let main_sock_for_task = state.main_sock.clone();
    tokio::spawn(async move {
        relay_task(alloc_for_task, main_sock_for_task, state_arc.config.server.software.clone()).await;
    });

    let mut resp = StunMessage::new(MSG_ALLOCATE_SUCCESS, msg.transaction_id);
    resp.add_xor_addr(ATTR_XOR_RELAYED_ADDRESS, relay_addr);
    resp.add_xor_addr(ATTR_XOR_MAPPED_ADDRESS, src);
    resp.add_u32(ATTR_LIFETIME, lifetime as u32);
    Some(resp.encode_with_integrity(Some(&mi_key), &state.config.server.software))
}

// ─── Refresh ──────────────────────────────────────────────────────────────────

fn handle_refresh(
    raw: &[u8],
    msg: StunMessage,
    src: SocketAddr,
    state: &Arc<TurnState>,
) -> Option<Vec<u8>> {
    let (username, mi_key) = match check_auth(raw, &msg, src, state, MSG_REFRESH_ERROR) {
        Ok(v)  => v,
        Err(e) => return Some(e),
    };

    let Some(alloc) = state.allocs.get(src) else {
        let mut err = StunMessage::new(MSG_REFRESH_ERROR, msg.transaction_id);
        err.add_error(437, "Allocation Mismatch");
        return Some(err.encode_simple(&state.config.server.software));
    };

    let requested = msg.get_attr(ATTR_LIFETIME)
        .filter(|b| b.len() == 4)
        .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as u64)
        .unwrap_or(state.config.limits.allocation_lifetime);

    if requested == 0 {
        // Lifetime 0 = delete allocation
        state.allocs.remove(src);
        info!("TURN dealloc: {}", src);
        let mut resp = StunMessage::new(MSG_REFRESH_SUCCESS, msg.transaction_id);
        resp.add_u32(ATTR_LIFETIME, 0);
        return Some(resp.encode_with_integrity(Some(&mi_key), &state.config.server.software));
    }

    let lifetime = requested.min(state.config.limits.allocation_lifetime);
    alloc.refresh(Duration::from_secs(lifetime));

    let mut resp = StunMessage::new(MSG_REFRESH_SUCCESS, msg.transaction_id);
    resp.add_u32(ATTR_LIFETIME, lifetime as u32);
    Some(resp.encode_with_integrity(Some(&mi_key), &state.config.server.software))
}

// ─── CreatePermission ─────────────────────────────────────────────────────────

fn handle_create_permission(
    raw: &[u8],
    msg: StunMessage,
    src: SocketAddr,
    state: &Arc<TurnState>,
) -> Option<Vec<u8>> {
    let (_, mi_key) = match check_auth(raw, &msg, src, state, 0x0118) {
        Ok(v)  => v,
        Err(e) => return Some(e),
    };

    let Some(alloc) = state.allocs.get(src) else {
        let mut err = StunMessage::new(0x0118, msg.transaction_id);
        err.add_error(437, "Allocation Mismatch");
        return Some(err.encode_simple(&state.config.server.software));
    };

    // All XOR-PEER-ADDRESS attributes specify peers to grant permission
    for (atype, aval) in &msg.attrs {
        if *atype == ATTR_XOR_PEER_ADDRESS {
            if let Some(peer_addr) = decode_xor_addr(aval, &msg.transaction_id) {
                if alloc.permission_count() < state.config.limits.max_permissions {
                    alloc.add_permission(peer_addr.ip());
                    debug!("TURN perm: {} ← {}", src, peer_addr.ip());
                }
            }
        }
    }

    let mut resp = StunMessage::new(MSG_CREATE_PERM_SUCCESS, msg.transaction_id);
    Some(resp.encode_with_integrity(Some(&mi_key), &state.config.server.software))
}

// ─── ChannelBind ──────────────────────────────────────────────────────────────

fn handle_channel_bind(
    raw: &[u8],
    msg: StunMessage,
    src: SocketAddr,
    state: &Arc<TurnState>,
) -> Option<Vec<u8>> {
    let (_, mi_key) = match check_auth(raw, &msg, src, state, 0x0119) {
        Ok(v)  => v,
        Err(e) => return Some(e),
    };

    let Some(alloc) = state.allocs.get(src) else {
        let mut err = StunMessage::new(0x0119, msg.transaction_id);
        err.add_error(437, "Allocation Mismatch");
        return Some(err.encode_simple(&state.config.server.software));
    };

    let channel = msg.get_attr(ATTR_CHANNEL_NUMBER)
        .filter(|b| b.len() >= 2)
        .map(|b| u16::from_be_bytes([b[0], b[1]]))?;

    if channel < 0x4000 || channel > 0x7FFF {
        let mut err = StunMessage::new(0x0119, msg.transaction_id);
        err.add_error(ERR_BAD_REQUEST.0, ERR_BAD_REQUEST.1);
        return Some(err.encode_with_integrity(Some(&mi_key), &state.config.server.software));
    }

    let peer = msg.get_attr(ATTR_XOR_PEER_ADDRESS)
        .and_then(|b| decode_xor_addr(b, &msg.transaction_id))?;

    if !alloc.bind_channel(channel, peer, state.config.limits.max_channels) {
        let mut err = StunMessage::new(0x0119, msg.transaction_id);
        err.add_error(ERR_BAD_REQUEST.0, "Channel already bound to different peer");
        return Some(err.encode_with_integrity(Some(&mi_key), &state.config.server.software));
    }
    // CreatePermission implicitly for the channel peer
    alloc.add_permission(peer.ip());

    debug!("TURN ch-bind: {} ch={:#06x} → {}", src, channel, peer);

    let mut resp = StunMessage::new(MSG_CHANNEL_BIND_SUCCESS, msg.transaction_id);
    Some(resp.encode_with_integrity(Some(&mi_key), &state.config.server.software))
}

// ─── Send indication (client → peer via relay) ────────────────────────────────

async fn handle_send_indication(msg: StunMessage, src: SocketAddr, state: &Arc<TurnState>) {
    let Some(alloc) = state.allocs.get(src) else { return; };

    let Some(peer_raw) = msg.get_attr(ATTR_XOR_PEER_ADDRESS) else { return; };
    let Some(peer) = decode_xor_addr(peer_raw, &msg.transaction_id) else { return; };
    let Some(data) = msg.get_attr(ATTR_DATA) else { return; };

    if !alloc.has_permission(peer.ip()) {
        warn!("TURN send: {} blocked (no perm for {})", src, peer.ip());
        return;
    }

    alloc.relay_socket.send_to(data, peer).await.ok();
}

// ─── ChannelData (client → peer via relay) ────────────────────────────────────

async fn handle_channel_data(
    raw: &[u8],
    src: SocketAddr,
    state: &Arc<TurnState>,
) -> Option<Vec<u8>> {
    let Some(alloc) = state.allocs.get(src) else { return None; };
    let ch = stun::parse_channel_data(raw)?;

    let Some(peer) = alloc.peer_for_channel(ch.channel) else { return None; };

    alloc.relay_socket.send_to(ch.data, peer).await.ok();
    None
}

// ─── Relay task (peer → client) ───────────────────────────────────────────────

/// Runs per allocation. Forwards packets arriving on the relay socket to the client.
async fn relay_task(
    alloc: Arc<Allocation>,
    main_sock: Arc<UdpSocket>,
    software: String,
) {
    let mut buf = vec![0u8; 65536];
    loop {
        if alloc.is_expired() {
            debug!("TURN relay task expired for {}", alloc.client_addr);
            return;
        }

        let (len, peer_addr) = match alloc.relay_socket.recv_from(&mut buf).await {
            Ok(v)  => v,
            Err(_) => return,
        };

        if !alloc.has_permission(peer_addr.ip()) {
            debug!("TURN relay: no permission for peer {} → {}", peer_addr, alloc.client_addr);
            continue;
        }

        // Check for channel binding first (more efficient)
        if let Some(channel) = alloc.channel_for_peer(peer_addr) {
            let pkt = encode_channel_data(channel, &buf[..len]);
            main_sock.send_to(&pkt, alloc.client_addr).await.ok();
        } else {
            // Fall back to Data indication
            let txid = [0u8; 12]; // indications use a fresh transaction ID
            let mut ind = StunMessage::new(MSG_DATA_INDICATION, txid);
            ind.add_xor_addr(ATTR_XOR_PEER_ADDRESS, peer_addr);
            ind.add_bytes(ATTR_DATA, &buf[..len]);
            let pkt = ind.encode_simple(&software);
            main_sock.send_to(&pkt, alloc.client_addr).await.ok();
        }
    }
}

// ─── Const used in check_auth for CreatePermission/ChannelBind error types ────
// (inline so the match arms in handle_create_permission / handle_channel_bind compile)
const ERR_INSUF_CAPACITY: (u16, &str) = (508, "Insufficient Capacity");
