//! Miner Link Relay Hub — WebSocket bridge between personal miners and wallets
//!
//! Architecture:
//!   Browser Wallet ←→ WS ←→ API Server (relay) ←→ WS ←→ Personal Miner
//!
//! The relay is keyed by wallet address. Each wallet gets a `WalletLinkHub`
//! with two broadcast channels (miner→wallet and wallet→miner). Messages
//! are forwarded bidirectionally without interpretation — the server is a
//! dumb pipe. This keeps latency minimal and avoids coupling the server
//! to the miner-link protocol evolution.

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    response::{IntoResponse, Response},
    Json,
};
use dashmap::DashMap;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

// ─── Registry ───────────────────────────────────────────────────────────────

/// Per-wallet hub that bridges miner↔wallet messages.
pub struct WalletLinkHub {
    /// miner → wallet direction
    pub miner_to_wallet_tx: broadcast::Sender<String>,
    /// wallet → miner direction
    pub wallet_to_miner_tx: broadcast::Sender<String>,
    /// Live count of connected miners
    pub miner_count: AtomicU32,
    /// Live count of connected wallets
    pub wallet_count: AtomicU32,
}

impl WalletLinkHub {
    fn new() -> Self {
        let (m2w_tx, _) = broadcast::channel(256);
        let (w2m_tx, _) = broadcast::channel(256);
        Self {
            miner_to_wallet_tx: m2w_tx,
            wallet_to_miner_tx: w2m_tx,
            miner_count: AtomicU32::new(0),
            wallet_count: AtomicU32::new(0),
        }
    }
}

/// Global registry mapping wallet address → hub. Wrapped in Arc for sharing.
pub type MinerLinkRegistry = Arc<DashMap<String, Arc<WalletLinkHub>>>;

pub fn new_registry() -> MinerLinkRegistry {
    Arc::new(DashMap::new())
}

// ─── Query params ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct WsQueryParams {
    pub role: String,      // "miner" or "wallet"
    pub wallet: String,    // wallet address (hex)
    #[serde(default)]
    pub miner_id: Option<String>,
}

// ─── WebSocket upgrade handler ──────────────────────────────────────────────

pub async fn miner_link_ws_handler(
    ws: WebSocketUpgrade,
    Query(params): Query<WsQueryParams>,
    State(state): State<Arc<crate::AppState>>,
) -> Response {
    let registry = state.miner_link_registry.clone();
    ws.on_upgrade(move |socket| handle_ws_connection(socket, params, registry))
}

async fn handle_ws_connection(
    socket: WebSocket,
    params: WsQueryParams,
    registry: MinerLinkRegistry,
) {
    let wallet = params.wallet.clone();
    let role = params.role.clone();

    // Get or create hub for this wallet
    let hub = registry
        .entry(wallet.clone())
        .or_insert_with(|| Arc::new(WalletLinkHub::new()))
        .clone();

    match role.as_str() {
        "miner" => {
            hub.miner_count.fetch_add(1, Ordering::Relaxed);
            info!(
                "🔗 [MinerLink] Miner connected for wallet {} (miner_id={:?}, total_miners={})",
                wallet,
                params.miner_id,
                hub.miner_count.load(Ordering::Relaxed)
            );
            handle_miner_session(socket, &hub, &wallet).await;
            let remaining = hub.miner_count.fetch_sub(1, Ordering::Relaxed) - 1;
            info!("🔗 [MinerLink] Miner disconnected for wallet {} (remaining={})", wallet, remaining);
        }
        "wallet" => {
            hub.wallet_count.fetch_add(1, Ordering::Relaxed);
            info!(
                "🔗 [MinerLink] Wallet connected for {} (total_wallets={})",
                wallet,
                hub.wallet_count.load(Ordering::Relaxed)
            );
            // Send link-established message
            let established = serde_json::json!({
                "type": "LinkEstablished",
                "peer_type": "wallet",
                "connected_miners": hub.miner_count.load(Ordering::Relaxed),
                "connected_wallets": hub.wallet_count.load(Ordering::Relaxed),
            });
            // We'll send this after the split, in the session handler
            handle_wallet_session(socket, &hub, &wallet, established.to_string()).await;
            let remaining = hub.wallet_count.fetch_sub(1, Ordering::Relaxed) - 1;
            info!("🔗 [MinerLink] Wallet disconnected for {} (remaining={})", wallet, remaining);
        }
        _ => {
            warn!("🔗 [MinerLink] Unknown role '{}', closing", role);
            return;
        }
    }

    // Clean up empty hubs
    let miners = hub.miner_count.load(Ordering::Relaxed);
    let wallets = hub.wallet_count.load(Ordering::Relaxed);
    if miners == 0 && wallets == 0 {
        registry.remove(&wallet);
        debug!("🔗 [MinerLink] Cleaned up empty hub for wallet {}", wallet);
    }
}

// ─── Miner session ──────────────────────────────────────────────────────────

async fn handle_miner_session(socket: WebSocket, hub: &WalletLinkHub, wallet: &str) {
    let (mut sink, mut stream) = socket.split();

    // Miner subscribes to wallet→miner channel (receives commands)
    let mut cmd_rx = hub.wallet_to_miner_tx.subscribe();
    // Miner publishes to miner→wallet channel (sends stats)
    let m2w_tx = hub.miner_to_wallet_tx.clone();

    let mut ping_interval = tokio::time::interval(tokio::time::Duration::from_secs(30));

    loop {
        tokio::select! {
            // Forward commands from wallet → miner
            cmd = cmd_rx.recv() => {
                match cmd {
                    Ok(text) => {
                        if sink.send(Message::Text(text)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        debug!("🔗 [MinerLink] Miner lagged {} messages for {}", n, wallet);
                    }
                    Err(_) => break,
                }
            }

            // Receive stats/acks from miner, relay to wallet
            msg = stream.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        // Relay to all connected wallets
                        let _ = m2w_tx.send(text);
                    }
                    Some(Ok(Message::Ping(data))) => {
                        let _ = sink.send(Message::Pong(data)).await;
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(_)) => break,
                    _ => {}
                }
            }

            // Keepalive ping
            _ = ping_interval.tick() => {
                if sink.send(Message::Ping(vec![])).await.is_err() {
                    break;
                }
            }
        }
    }
}

// ─── Wallet session ─────────────────────────────────────────────────────────

async fn handle_wallet_session(socket: WebSocket, hub: &WalletLinkHub, wallet: &str, welcome: String) {
    let (mut sink, mut stream) = socket.split();

    // Send welcome/link-established
    let _ = sink.send(Message::Text(welcome)).await;

    // Wallet subscribes to miner→wallet channel (receives stats)
    let mut stats_rx = hub.miner_to_wallet_tx.subscribe();
    // Wallet publishes to wallet→miner channel (sends commands)
    let w2m_tx = hub.wallet_to_miner_tx.clone();

    let mut ping_interval = tokio::time::interval(tokio::time::Duration::from_secs(30));

    loop {
        tokio::select! {
            // Forward stats from miner → wallet
            stat = stats_rx.recv() => {
                match stat {
                    Ok(text) => {
                        if sink.send(Message::Text(text)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        debug!("🔗 [MinerLink] Wallet lagged {} messages for {}", n, wallet);
                    }
                    Err(_) => break,
                }
            }

            // Receive commands from wallet, relay to miner
            msg = stream.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        // Relay command to all connected miners
                        let _ = w2m_tx.send(text);
                    }
                    Some(Ok(Message::Ping(data))) => {
                        let _ = sink.send(Message::Pong(data)).await;
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(_)) => break,
                    _ => {}
                }
            }

            // Keepalive ping
            _ = ping_interval.tick() => {
                if sink.send(Message::Ping(vec![])).await.is_err() {
                    break;
                }
            }
        }
    }
}

// ─── REST status endpoint ───────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct MinerLinkStatus {
    pub wallet: String,
    pub connected_miners: u32,
    pub connected_wallets: u32,
}

pub async fn get_link_status(
    axum::extract::Path(wallet): axum::extract::Path<String>,
    State(state): State<Arc<crate::AppState>>,
) -> Json<serde_json::Value> {
    let registry = &state.miner_link_registry;
    if let Some(hub) = registry.get(&wallet) {
        Json(serde_json::json!({
            "success": true,
            "data": {
                "wallet": wallet,
                "connected_miners": hub.miner_count.load(Ordering::Relaxed),
                "connected_wallets": hub.wallet_count.load(Ordering::Relaxed),
            }
        }))
    } else {
        Json(serde_json::json!({
            "success": true,
            "data": {
                "wallet": wallet,
                "connected_miners": 0,
                "connected_wallets": 0,
            }
        }))
    }
}
