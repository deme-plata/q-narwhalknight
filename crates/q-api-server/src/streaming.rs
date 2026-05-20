/// Real-time streaming support for Q-NarwhalKnight
/// Provides both SSE (Server-Sent Events) and WebSocket streaming
/// Target latency: <50ms for critical updates
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::{header, StatusCode},
    response::{
        sse::{Event, Sse},
        Response,
    },
};
use axum_extra::{headers, TypedHeader};
use futures_util::{sink::SinkExt, stream::StreamExt as FuturesStreamExt};
use q_storage::BalanceStorage; // Import trait for get_balance method
use q_types::*;
use serde_json;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio_stream::{wrappers::BroadcastStream, StreamExt as TokioStreamExt};
use tracing::{debug, error, info, trace, warn};

use axum::extract::Query;
use crate::AppState;
use crate::wallet_auth::AuthenticatedWallet;

/// v1.0.2: SSE query parameters for bandwidth optimization
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SseQueryParams {
    /// Filter events to a single wallet address
    pub wallet_address: Option<String>,
    /// When true, NewBlock events send compact headers (~100 bytes) instead of full block data (~2-5KB)
    #[serde(default)]
    pub headers_only: bool,
    /// v9.0.1: When true, only forward mining-relevant events (NewBlock, MiningReward,
    /// BalanceUpdated, PendingMiningReward, MiningStats). Drops all noise events
    /// (MetricsUpdate, NodeStatusUpdate, TokenPriceUpdate, emails, calendar, etc.)
    /// reducing miner bandwidth from ~111 KB/s to ~2-5 KB/s.
    #[serde(default)]
    pub miner_mode: bool,
    /// NEW v10.9.58: signed JSON auth passed as URL query (for browser EventSource clients
    /// that cannot set X-Wallet-Auth header). Same JSON format as the header.
    pub auth: Option<String>,
}

/// v1.0.2: WebSocket query parameters
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WsQueryParams {
    /// Filter events to a single wallet address
    pub wallet_address: Option<String>,
    /// When true, NewBlock events send compact headers only
    #[serde(default)]
    pub headers_only: bool,
}

/// v1.2.0-beta Phase 3: Default confirmation status for balance updates
/// Used when deserializing older events without the confirmation_status field
fn default_confirmation_status() -> String {
    "instant".to_string()
}

/// Real-time events that can be streamed to clients
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", content = "data")]
pub enum StreamEvent {
    /// New transaction submitted to mempool
    TransactionSubmitted {
        transaction: Transaction,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Transaction status updated
    TransactionStatusUpdate {
        tx_hash: TxHash,
        old_status: TxStatus,
        new_status: TxStatus,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// New vertex created in DAG
    VertexCreated {
        vertex: Vertex,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// New certificate generated
    CertificateGenerated {
        certificate: Certificate,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Block finalized
    BlockFinalized {
        height: Height,
        round: Round,
        transactions: Vec<TxHash>,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// New block produced
    NewBlock {
        height: u64,
        hash: String,
        prev_hash: String,
        solutions_count: usize,
        total_difficulty: u128,
        dag_round: u64,
        miner_count: usize,
        tx_count: usize,
        block_reward: f64,
        producer_id: usize, // Parallel producer ID for lane assignment in visualization
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Node status update
    NodeStatusUpdate {
        status: NodeStatus,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Peer connection events
    PeerEvent {
        peer_id: String,
        event_type: PeerEventType,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Performance metrics update
    MetricsUpdate {
        throughput: u64, // tx/s
        latency_ms: u64,
        cpu_usage: f32,
        memory_usage: u64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Peer discovered through Bitcoin network
    PeerDiscovered {
        node_id: String,
        confidence: f64,
        method: String, // "bitcoin", "dns-phantom"
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Peer connected
    PeerConnected {
        node_id: String,
        connection_type: String, // "bitcoin-tor", "direct"
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Peer disconnected
    PeerDisconnected {
        node_id: String,
        reason: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// DNS-Phantom peer discovered
    PhantomPeerDiscovered {
        node_id: String,
        discovery_method: String,
        confidence: f64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// DNS-Phantom message received
    PhantomMessageReceived {
        from: String,
        message_type: String,
        size_bytes: usize,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Security alert
    SecurityAlert {
        alert_type: String,
        description: String,
        risk_level: f64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Network topology changed
    NetworkTopologyChanged {
        total_peers: u32,
        direct_peers: u32,
        phantom_peers: u32,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Tor circuit event
    TorCircuitEvent {
        circuit_id: u32,
        event_type: String, // "built", "failed", "closed"
        details: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    // v7.0.0: FaucetDispensed removed — faucet eliminated
    /// Wallet balance updated (for transaction processing)
    /// v1.2.0-beta Phase 3: Enhanced with block tracking and confirmation status
    BalanceUpdated {
        wallet_address: String,
        old_balance: f64,
        new_balance: f64,
        change_reason: String, // "transaction_sent", "transaction_received", "faucet", "coinbase_reward", "p2p_mining_reward"
        timestamp: chrono::DateTime<chrono::Utc>,
        /// v1.2.0-beta Phase 3: Block hash where this update was confirmed (hex string)
        /// None for instant updates (faucet, P2P sync) that haven't been included in a block yet
        #[serde(skip_serializing_if = "Option::is_none")]
        block_hash: Option<String>,
        /// v1.2.0-beta Phase 3: Block height where this update was confirmed
        #[serde(skip_serializing_if = "Option::is_none")]
        block_height: Option<u64>,
        /// v1.2.0-beta Phase 3: Confirmation status
        /// - "pending": Update received via gossipsub, waiting for block inclusion
        /// - "confirmed": Update included in a finalized block via DAG-Knight consensus
        /// - "instant": Immediate local update (faucet, debugging)
        #[serde(default = "default_confirmation_status")]
        confirmation_status: String,
        /// Sender address (hex) — populated for transaction_received events
        #[serde(skip_serializing_if = "Option::is_none")]
        from_address: Option<String>,
        /// Transaction hash (hex) — populated for transaction_received events
        #[serde(skip_serializing_if = "Option::is_none")]
        tx_hash: Option<String>,
        /// Memo attached to the transaction — populated when sender included one
        #[serde(skip_serializing_if = "Option::is_none")]
        memo: Option<String>,
    },
    /// v1.4.10-beta: Custom token balance updated (for instant DEX updates)
    TokenBalanceUpdated {
        wallet_address: String,
        token_address: String,
        token_symbol: String,
        old_balance: f64,
        new_balance: f64,
        change_reason: String, // "transfer_sent", "transfer_received", "swap", "mint", "burn"
        timestamp: chrono::DateTime<chrono::Utc>,
        #[serde(skip_serializing_if = "Option::is_none")]
        block_hash: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        block_height: Option<u64>,
        #[serde(default = "default_confirmation_status")]
        confirmation_status: String,
    },
    /// Privacy mixing started
    PrivacyMixingStarted {
        transaction_hash: TxHash,
        mixing_session_id: String,
        privacy_level: String,
        decoy_count: u32,
        estimated_completion_seconds: u32,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Privacy mixing completed
    PrivacyMixingCompleted {
        transaction_hash: TxHash,
        mixing_session_id: String,
        final_anonymity_set_size: u32,
        mixing_duration_seconds: u32,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Nitro boost applied to a token
    NitroBoost {
        token_id: String,
        points: u64,
        total_points: u64,
        boosted_by: String, // wallet address
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Bulk update of all nitro boost points
    NitroBoostsUpdate {
        boosts: std::collections::HashMap<String, u64>, // token_id -> total_points
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Token price update from swap or oracle
    /// v2.9.22-beta: Added change_1h and change_7d for full price metrics
    /// v2.9.25-beta: Added token_address for frontend matching by address
    TokenPriceUpdate {
        token_symbol: String,
        token_address: Option<String>,  // 🆕 v2.9.25-beta: Token contract address for matching
        price: f64,
        change_1h: f64,   // 🆕 v2.9.22-beta: 1-hour price change percentage
        change_24h: f64,
        change_7d: f64,   // 🆕 v2.9.22-beta: 7-day price change percentage
        volume_24h: f64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Liquidity pool update
    LiquidityPoolUpdate {
        pool_id: String,
        token0: String,
        token1: String,
        reserve0: u64,
        reserve1: u64,
        total_liquidity: u64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Swap executed event
    SwapExecuted {
        from_token: String,
        to_token: String,
        amount_in: u128,
        amount_out: u128,
        wallet_address: String,
        price_impact: f64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Mining reward earned event
    MiningReward {
        miner_address: String,
        reward_qnk: f64,
        nonce: u64,
        block_height: u64,
        difficulty: String,
        hash_rate: f64,
        miner_id: Option<String>, // 🆕 v3.3.3-beta: Unique miner instance ID for identification
        worker_name: Option<String>, // 🆕 v0.6.2-beta: Human-readable miner name (e.g., "Server Alpha")
        origin_node_id: Option<String>, // 🆕 v2.3.5-beta: Which node mined this reward (peer ID)
        origin_node_name: Option<String>, // 🆕 v2.3.5-beta: Human-friendly node name (e.g., "Bootstrap", "Alpha")
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Mining statistics update
    /// v3.2.25-beta: Added miner_id and worker_id to distinguish multiple miners to same wallet
    MiningStats {
        miner_address: String,
        total_rewards: f64,
        total_blocks_found: u64,
        current_balance: f64,
        avg_hash_rate: f64,
        /// v3.2.25-beta: Unique miner instance ID (from miner software)
        miner_id: Option<String>,
        /// v3.2.25-beta: Worker identifier (miner_id, worker_name, or "direct"/"p2p:NODE")
        worker_id: Option<String>,
        /// v7.4.2: Human-readable miner name from --miner-name CLI arg
        worker_name: Option<String>,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// v1.3.8-beta: Pending mining reward from P2P gossip
    /// This provides instant UI feedback for users mining to localhost
    /// while frontend is connected to bootstrap node.
    /// NOTE: This is for UI display ONLY - not consensus-confirmed!
    PendingMiningReward {
        miner_address: String,
        /// Pending reward amount in QNK (8 decimals)
        pending_reward_qnk: f64,
        /// Node that mined this reward (for attribution)
        origin_node_id: String,
        /// Block height at the mining node
        source_height: u64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Custom event for mining rewards and other custom types
    Custom {
        event_type: String,
        data: serde_json::Value,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// v1.4.3: QNO Oracle data update
    QnoOracleUpdate {
        domain: String,
        value: f64,
        confidence: f64,
        sources: Vec<OracleSourceInfo>,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// v1.4.3: QNO prediction resolution result
    QnoResolution {
        stake_id: String,
        domain: String,
        predicted_value: f64,
        actual_value: f64,
        accuracy_score: f64,
        is_accurate: bool,
        slashing_applied: f64,
        reward_adjustment: f64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// v1.4.3: QNO new stake placed
    QnoStake {
        stake_id: String,
        domain: String,
        amount: f64,
        confidence: f64,
        prediction_value: f64,
        wallet_address: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// v1.4.3: QNO slashing event
    QnoSlashing {
        stake_id: String,
        domain: String,
        amount_slashed: f64,
        reason: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// v5.6.0: Server version broadcast after startup/restart
    /// Clients compare with cached version to show refresh banner
    ServerVersion {
        version: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// v5.2.0: State sync completed — frontend should refresh contracts/pools/balances
    StateSyncComplete {
        contracts_added: usize,
        pools_added: usize,
        balances_added: usize,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// v5.7.0: Mining pool statistics updated
    PoolStatsUpdated {
        hashrate: f64,
        workers: usize,
        blocks_found: u64,
        current_round: u64,
        difficulty: f64,
        total_shares: u64,
        valid_shares: u64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// v5.7.0: Mining pool share accepted
    PoolShareAccepted {
        worker: String,
        difficulty: f64,
        share_id: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// v5.7.0: Mining pool block found
    PoolBlockFound {
        block_hash: String,
        height: u64,
        finder: String,
        reward: f64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// v5.7.0: Mining pool payout event
    PoolPayout {
        batch_id: String,
        total_reward: f64,
        payout_count: usize,
        status: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// v7.3.2: Email received (wallet-to-wallet or SMTP inbound)
    EmailReceived {
        email_id: String,
        from_address: String,
        subject: String,
        preview: String,
        has_crypto: bool,
        crypto_amount: Option<f64>,
        crypto_token: Option<String>,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// v7.3.2: Email sent confirmation
    EmailSent {
        email_id: String,
        to_address: String,
        subject: String,
        delivery_method: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// v7.3.2: Unread email count update
    EmailUnreadCount {
        count: u64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// v7.3.3: Calendar event created/updated
    CalendarEventCreated {
        event_id: String,
        title: String,
        event_type: String,
        start_time: u64,
        has_scheduled_tx: bool,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// v7.3.3: Calendar reminder notification
    CalendarReminder {
        event_id: String,
        title: String,
        event_type: String,
        minutes_until: i64,
        start_time: u64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// v7.3.3: Scheduled transaction executed
    ScheduledTransactionExecuted {
        event_id: String,
        title: String,
        to_wallet: String,
        token: String,
        amount: String,
        tx_hash: Option<String>,
        success: bool,
        error: Option<String>,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// v9.1.4: Admin-triggered mining mode switch — broadcast to all connected miners
    MiningModeSwitch {
        target_mode: String,
        pool_url: Option<String>,
        reason: Option<String>,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// v9.5.0: Starship Endgame — Compute orchestrator status update
    ComputeStatus {
        mode: String,
        ai_inference: Option<serde_json::Value>,
        layers: Vec<serde_json::Value>,
        trainer_active: bool,
        performance_boost_pct: f32,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// v10.2.0: Crown & Ash — Game tick completed (1 tick = 10 blocks)
    GameTick {
        turn: u32,
        events_count: usize,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// v10.2.0: Crown & Ash — Player action processed by simulation
    GameActionProcessed {
        wallet: String,
        action_type: String,
        turn: u32,
        success: bool,
        message: Option<String>,
    },

    /// v10.2.0: Crown & Ash — Full world state snapshot (sent on connect + periodically)
    GameStateSnapshot {
        turn: u32,
        province_count: u16,
        faction_count: u8,
        player_count: u8,
        data: serde_json::Value,
    },

    /// v10.2.0: Crown & Ash — Narrative game event (battle, death, plague, etc.)
    GameEvent {
        turn: u32,
        event_type: String,
        description: String,
        data: serde_json::Value,
    },
}

/// v1.4.3: Oracle source information for SSE events
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OracleSourceInfo {
    pub provider: String,
    pub value: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub enum PeerEventType {
    Connected,
    Disconnected,
    MessageReceived,
    MessageSent,
}

/// Event broadcaster for managing real-time streams
pub struct EventBroadcaster {
    tx: broadcast::Sender<StreamEvent>,
    // Deduplication cache: stores (wallet_address, balance) with timestamp to prevent duplicate broadcasts
    recent_balance_broadcasts:
        Arc<tokio::sync::Mutex<std::collections::HashMap<String, (f64, std::time::Instant)>>>,
}

impl EventBroadcaster {
    pub fn new() -> Self {
        // v6.0.10: Reduced from 100k to 10k to save ~45MB RAM.
        // 100k × ~500 bytes/event = 50MB buffer that's rarely needed.
        // 10k is sufficient for real-time SSE delivery.
        let (tx, _rx) = broadcast::channel(10_000);
        Self {
            tx,
            recent_balance_broadcasts: Arc::new(tokio::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
        }
    }

    /// Broadcast an event to all subscribers
    pub async fn broadcast(
        &self,
        event: StreamEvent,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let subscriber_count = self.tx.receiver_count();

        // 🔒 DEDUPLICATION: Skip duplicate balance updates within 100ms window
        // v9.2.7: Reduced from 500ms to 100ms for faster SSE updates
        if let StreamEvent::BalanceUpdated {
            wallet_address,
            new_balance,
            ..
        } = &event
        {
            let mut cache = self.recent_balance_broadcasts.lock().await;
            let now = std::time::Instant::now();

            // Check if we recently broadcast this exact balance
            if let Some((last_balance, last_time)) = cache.get(wallet_address) {
                if (*last_balance - new_balance).abs() < 0.00000001
                    && now.duration_since(*last_time).as_millis() < 100
                {
                    trace!(
                        "📡 [SSE] Skipping duplicate BalanceUpdated for {}... (within 100ms)",
                        &wallet_address[..16]
                    );
                    return Ok(());
                }
            }

            // Update cache
            cache.insert(wallet_address.clone(), (*new_balance, now));

            // Clean old entries (older than 500ms)
            cache.retain(|_, (_, time)| now.duration_since(*time).as_millis() < 500);
        }

        // 🔒 PRIVACY: Log aggregate statistics only, no individual wallet data
        match &event {
            StreamEvent::BalanceUpdated { change_reason, .. } => {
                // v3.4.2: Reduced to trace to prevent log spam
                trace!(
                    "📡 [SSE] Broadcasting BalanceUpdated: reason={}, subscribers={}",
                    change_reason, subscriber_count
                );
            }
            StreamEvent::PrivacyMixingCompleted { .. } => {
                // v6.0.2: No identifying info in privacy transaction logs
                debug!("📡 [SSE] Broadcasting transaction completion, subscribers={}", subscriber_count);
            }
            _ => {
                // v3.4.4: Reduced to trace to prevent log spam
                trace!(
                    "Broadcasting event: {}, subscriber count: {}",
                    event_type_name(&event),
                    subscriber_count
                );
            }
        }

        // Only send if there are active subscribers to avoid "channel closed" errors
        if subscriber_count > 0 {
            match self.tx.send(event) {
                Ok(_) => {
                    debug!(
                        "Event broadcast successful to {} subscribers",
                        subscriber_count
                    );
                    Ok(())
                }
                Err(e) => {
                    warn!("Event broadcast failed: {}", e);
                    Err(e)
                }
            }
        } else {
            // Silently succeed when no subscribers are present (trace level to reduce spam)
            trace!("Event emission skipped: no active subscribers");
            Ok(())
        }
    }

    /// Subscribe to events
    pub fn subscribe(&self) -> broadcast::Receiver<StreamEvent> {
        let receiver = self.tx.subscribe();
        let new_count = self.tx.receiver_count();
        debug!("New subscriber added, total subscribers: {}", new_count);
        receiver
    }

    /// Get current subscriber count
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

/// SSE endpoint for real-time event streaming with privacy filtering.
///
/// Usage: GET /api/v1/events?wallet_address=<address>&headers_only=true&miner_mode=true
///
/// v11.0.0 / v11.1.0 PRIVACY HARDENING:
/// - **No wallet filter** → subscriber receives **public events only**
///   (NodeStatusUpdate, BlockFinalized, NewBlock, MetricsUpdate, TokenPriceUpdate,
///    LiquidityPoolUpdate, NitroBoostsUpdate, ServerVersion, StateSyncComplete).
///   Previously this returned ALL events (broadcast leak — anyone reading any wallet's
///   BalanceUpdated/MiningReward/EmailReceived/etc.).
/// - **`?wallet_address=W` requires X-Wallet-Auth proving ownership of W.** On mismatch
///   or missing header, the request is rejected with HTTP 401 (v11.1.1 hardening — was
///   previously a silent downgrade to the public stream, which masked auth failures
///   from browser clients).
///   Bad signatures are also rejected by the AuthenticatedWallet extractor (HTTP 401).
/// - **Email and Calendar events** are dropped from per-wallet SSE streams because the
///   StreamEvent schema currently lacks an owner-address field that would let the server
///   filter them. Frontends that rely on Email/Calendar SSE must wait for the schema to
///   carry the owner address (see TODO in is_event_relevant).
///
/// TODO(privacy): browser EventSource cannot send custom headers. For browser frontends,
/// add a `?auth=<base64url(auth_blob)>` query-parameter path that reuses the same payload
/// the X-Wallet-Auth header carries. Until that lands, browser clients should subscribe
/// via fetch+ReadableStream (or a polyfill) so they can attach the header.
///
/// v1.0.2 Bandwidth optimizations:
/// - `?headers_only=true` — NewBlock events send compact headers (~100 bytes vs ~2-5KB)
/// v9.0.1 Bandwidth optimizations:
/// - `?miner_mode=true` — Only forward mining-relevant events (NewBlock, MiningReward,
///   BalanceUpdated, PendingMiningReward, MiningStats). Drops all noise events
///   (MetricsUpdate, NodeStatusUpdate, TokenPriceUpdate, emails, calendar, etc.)
///   reducing miner bandwidth from ~111 KB/s to ~2-5 KB/s.
pub async fn sse_events(
    State(state): State<Arc<AppState>>,
    user_agent: Option<TypedHeader<headers::UserAgent>>,
    Query(params): Query<SseQueryParams>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, axum::Error>>>, (StatusCode, &'static str)> {
    let user_agent_str = user_agent
        .as_ref()
        .map(|ua| ua.as_str())
        .unwrap_or("rust-client");
    info!("New SSE client connected: {} (headers_only={}, miner_mode={})", user_agent_str, params.headers_only, params.miner_mode);

    // Extract wallet_address filter parameter (optional)
    let requested_filter = params.wallet_address.clone();
    let headers_only = params.headers_only;
    let miner_mode = params.miner_mode;
    let auth_wallet = match auth_wallet {
        Some(w) => Some(w),
        None => {
            if let Some(auth_json) = params.auth.as_ref() {
                match crate::wallet_auth::validate_wallet_auth_query(auth_json, "/api/v1/events") {
                    Ok(addr) => Some(AuthenticatedWallet {
                        address: addr,
                        timestamp: chrono::Utc::now(),
                        scheme: crate::wallet_auth::AuthScheme::Ed25519,
                    }),
                    Err(e) => {
                        warn!("SSE query auth failed: {}", e);
                        return Err((StatusCode::UNAUTHORIZED, "query auth invalid"));
                    }
                }
            } else {
                None
            }
        }
    };

    // v11.1.1 PRIVACY: REQUIRE X-Wallet-Auth for any ?wallet_address= filter.
    // Earlier versions silently downgraded missing/mismatched auth to a public stream,
    // which let browser clients silently lose wallet-specific events without knowing
    // why. v11.1.1 returns 401 instead — explicit failure beats invisible degradation.
    // Unfiltered public streams still work without auth.
    let wallet_filter: Option<String> = match (requested_filter.as_ref(), auth_wallet.as_ref()) {
        (Some(filter), Some(auth)) => {
            let hex_part = filter.strip_prefix("qnk").unwrap_or(filter.as_str());
            let matches = hex_part.len() == 64
                && hex::decode(hex_part)
                    .ok()
                    .filter(|b| b.len() == 32)
                    .map(|b| {
                        let mut arr = [0u8; 32];
                        arr.copy_from_slice(&b);
                        arr == auth.address
                    })
                    .unwrap_or(false);
            if matches {
                requested_filter.clone()
            } else {
                warn!("🔒 SSE: wallet_filter does not match authenticated wallet — rejecting with 401");
                return Err((StatusCode::UNAUTHORIZED, "wallet_address does not match X-Wallet-Auth"));
            }
        }
        (Some(_filter), None) => {
            warn!(
                "🔒 SSE: wallet_filter requested without X-Wallet-Auth — rejecting with 401 \
                 (v11.1.1 hardening; supply X-Wallet-Auth header to receive wallet-specific events)"
            );
            return Err((StatusCode::UNAUTHORIZED, "X-Wallet-Auth required for wallet_address filter"));
        }
        (None, _) => None,
    };

    if let Some(ref wallet) = wallet_filter {
        // 🔒 PRIVACY: Hash wallet address for logging
        use blake3::hash;
        let wallet_hash = hash(wallet.as_bytes());
        debug!(
            "🔐 SSE connection established: wallet_hash={}",
            hex::encode(&wallet_hash.as_bytes()[..8])
        );
    } else {
        debug!("SSE connection without wallet filter — public events only (v11.0.0 hardening)");
    }

    // Create a manual stream that keeps the receiver alive
    let rx = state.event_broadcaster.subscribe();
    let broadcaster_clone = state.event_broadcaster.clone();

    // Log initial subscriber count
    debug!(
        "SSE subscriber count after connection: {}",
        broadcaster_clone.subscriber_count()
    );

    // Helper function to check if event is relevant to the wallet
    let is_event_relevant = move |event: &StreamEvent, filter: &Option<String>| -> bool {
        // v11.0.0 PRIVACY: No filter → only public-by-nature events. Previously returned
        // true for all events, which exposed every wallet's BalanceUpdated, MiningReward,
        // EmailReceived, etc. to any unfiltered subscriber.
        let Some(wallet_addr) = filter else {
            return matches!(event,
                StreamEvent::NodeStatusUpdate { .. }
                | StreamEvent::BlockFinalized { .. }
                | StreamEvent::NewBlock { .. }
                | StreamEvent::MetricsUpdate { .. }
                | StreamEvent::TokenPriceUpdate { .. }
                | StreamEvent::LiquidityPoolUpdate { .. }
                | StreamEvent::NitroBoostsUpdate { .. }
                | StreamEvent::ServerVersion { .. }
                | StreamEvent::StateSyncComplete { .. }
            );
        };

        // Normalize wallet address (remove "qnk" prefix if present)
        let normalized_filter = if wallet_addr.starts_with("qnk") {
            wallet_addr[3..].to_string()
        } else {
            wallet_addr.clone()
        };

        match event {
            // Transaction events - filter by from or to address
            StreamEvent::TransactionSubmitted { transaction, .. } => {
                // Check if wallet is sender or receiver
                let from_hex = hex::encode(&transaction.from);
                let to_hex = hex::encode(&transaction.to);

                from_hex == normalized_filter || to_hex == normalized_filter
            }

            // Transaction status updates - need to check transaction details
            // For now, allow all status updates (they're small events)
            StreamEvent::TransactionStatusUpdate { .. } => true,

            // Balance updates - only send if it's for this wallet
            StreamEvent::BalanceUpdated { wallet_address, change_reason, old_balance, new_balance, .. } => {
                let normalized_event = if wallet_address.starts_with("qnk") {
                    wallet_address[3..].to_string()
                } else {
                    wallet_address.clone()
                };
                let matches = normalized_event == normalized_filter;
                // v2.7.8-beta: Extensive debugging for P2P balance propagation (trace level to avoid log spam)
                trace!("🔍 [SSE FILTER] BalanceUpdated: reason={}, event_addr={} (first 16), filter={} (first 16), old={:.8}, new={:.8}, matches={}",
                      change_reason,
                      &normalized_event[..16.min(normalized_event.len())],
                      &normalized_filter[..16.min(normalized_filter.len())],
                      old_balance,
                      new_balance,
                      matches);
                matches
            }

            // Mining rewards - only send if it's for this wallet
            StreamEvent::MiningReward { miner_address, .. } => {
                let normalized_event = if miner_address.starts_with("qnk") {
                    miner_address[3..].to_string()
                } else {
                    miner_address.clone()
                };
                let matches = normalized_event == normalized_filter;
                // v3.3.4-beta: Debug logging for SSE filter (trace level to avoid log spam)
                trace!("🔍 [SSE FILTER] MiningReward: event_addr={} (len={}), filter={} (len={}), matches={}",
                      &normalized_event[..16.min(normalized_event.len())],
                      normalized_event.len(),
                      &normalized_filter[..16.min(normalized_filter.len())],
                      normalized_filter.len(),
                      matches);
                matches
            }

            // Mining stats - only send if it's for this wallet
            StreamEvent::MiningStats { miner_address, .. } => {
                let normalized_event = if miner_address.starts_with("qnk") {
                    miner_address[3..].to_string()
                } else {
                    miner_address.clone()
                };
                let matches = normalized_event == normalized_filter;
                // v2.7.6-beta: Debug logging for SSE filter (trace level to avoid log spam)
                trace!("🔍 [SSE FILTER] MiningStats: event_addr_len={}, filter_len={}, event_prefix={}, filter_prefix={}, matches={}",
                      normalized_event.len(),
                      normalized_filter.len(),
                      &normalized_event[..16.min(normalized_event.len())],
                      &normalized_filter[..16.min(normalized_filter.len())],
                      matches);
                matches
            }

            // v1.3.8-beta: Pending mining reward - only send if it's for this wallet
            StreamEvent::PendingMiningReward { miner_address, .. } => {
                let normalized_event = if miner_address.starts_with("qnk") {
                    miner_address[3..].to_string()
                } else {
                    miner_address.clone()
                };
                let matches = normalized_event == normalized_filter;
                // v2.7.6-beta: Debug logging for SSE filter (trace level to avoid log spam)
                trace!("🔍 [SSE FILTER] PendingMiningReward: event_addr={} (first 16), filter={} (first 16), matches={}",
                      &normalized_event[..16.min(normalized_event.len())],
                      &normalized_filter[..16.min(normalized_filter.len())],
                      matches);
                matches
            }

            // Swap events - only send if it's for this wallet
            StreamEvent::SwapExecuted { wallet_address, .. } => {
                let normalized_event = if wallet_address.starts_with("qnk") {
                    wallet_address[3..].to_string()
                } else {
                    wallet_address.clone()
                };
                normalized_event == normalized_filter
            }

            // Public events that everyone should see
            StreamEvent::NodeStatusUpdate { .. }
            | StreamEvent::BlockFinalized { .. }
            | StreamEvent::NewBlock { .. }
            | StreamEvent::MetricsUpdate { .. }
            | StreamEvent::TokenPriceUpdate { .. }
            | StreamEvent::LiquidityPoolUpdate { .. }
            | StreamEvent::NitroBoostsUpdate { .. }
            | StreamEvent::ServerVersion { .. }
            | StreamEvent::StateSyncComplete { .. } => true,

            // v11.0.0 PRIVACY: Email and Calendar StreamEvent variants currently lack an
            // owner-address field in their schema (see definitions at L427–479), so the
            // server cannot match them to a per-subscriber filter. Dropping them closes
            // the broadcast-to-all leak (previously: `=> true` shipped every user's email
            // and calendar events to every SSE subscriber, with only frontend-side filter).
            //
            // TODO(privacy): add `owner_address: String` to EmailReceived, EmailSent,
            // EmailUnreadCount, CalendarEventCreated, CalendarReminder,
            // ScheduledTransactionExecuted. Update emit sites (email_api.rs:905 et al.)
            // and the calendar emit sites. Then filter by `owner_address == normalized_filter`.
            StreamEvent::EmailReceived { .. }
            | StreamEvent::EmailSent { .. }
            | StreamEvent::EmailUnreadCount { .. }
            | StreamEvent::CalendarEventCreated { .. }
            | StreamEvent::CalendarReminder { .. }
            | StreamEvent::ScheduledTransactionExecuted { .. } => false,

            // v10.2.9: Token balance updates (QUGUSD, custom tokens) — filter by wallet address
            StreamEvent::TokenBalanceUpdated { ref wallet_address, .. } => {
                let normalized_event = if wallet_address.starts_with("qnk") {
                    wallet_address[3..].to_string()
                } else {
                    wallet_address.clone()
                };
                normalized_event == normalized_filter
            }

            // All other events are private - filter them out
            _ => false,
        }
    };

    // Clone state for initial balance fetch
    let state_clone = state.clone();
    let wallet_filter_clone = wallet_filter.clone();

    // v9.8.4: Max SSE connection lifetime — prevents unbounded accumulation
    let max_sse_lifetime_secs: u64 = std::env::var("MAX_SSE_LIFETIME_SECS")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(600); // 10 minutes default
    let connection_start = std::time::Instant::now();
    let max_sse_lifetime = std::time::Duration::from_secs(max_sse_lifetime_secs);

    let stream = futures_util::stream::unfold(
        (rx, wallet_filter, Some(state_clone), wallet_filter_clone, headers_only, miner_mode, connection_start, max_sse_lifetime),
        move |(mut rx, filter, state_opt, wallet_filter_for_initial, headers_only, miner_mode, connection_start, max_sse_lifetime)| async move {
            // CRITICAL FIX: Send initial balance event on SSE connection
            // This eliminates the "wait minutes for balance" issue
            if let (Some(state), Some(ref wallet_filter_value)) =
                (&state_opt, &wallet_filter_for_initial)
            {
                debug!("📡 SSE: Sending initial balance");

                // Use RocksDB as authoritative source for the initial SSE balance event.
                // The in-memory wallet_balances HashMap starts EMPTY on node restart and takes
                // 15+ seconds to populate. Using it caused initial SSE events to send balance=0,
                // which overwrote the user's correct locally-cached balance with 0, then jumped
                // to the real balance once the 15s sync ran — confusing users with balance spikes.
                // RocksDB is always authoritative and available immediately on startup.
                let wallet_hex = wallet_filter_value
                    .strip_prefix("qnk")
                    .unwrap_or(wallet_filter_value);

                // Always read from RocksDB for initial event (authoritative, available at startup)
                // Fall back to in-memory cache only if RocksDB read fails
                let balance = {
                    let rocksdb_balance = state.storage_engine.get_balance(wallet_hex).await.unwrap_or(0);
                    if rocksdb_balance > 0 {
                        rocksdb_balance
                    } else if let Ok(addr_bytes) = hex::decode(wallet_hex) {
                        if addr_bytes.len() == 32 {
                            let mut addr_array = [0u8; 32];
                            addr_array.copy_from_slice(&addr_bytes);
                            let balances = state.wallet_balances.read().await;
                            balances.get(&addr_array).copied().unwrap_or(0)
                        } else { 0 }
                    } else { 0 }
                };

                // v3.0.6-beta FIX: Use 1e24 divisor for u128 migration (was 100_000_000.0)
                let balance_qnk = balance as f64 / 1e24;
                // 🔒 PRIVACY: No logging of balances or addresses
                debug!("💰 SSE: Initial balance fetched from RocksDB");

                // Create initial balance event
                let initial_balance_event = serde_json::json!({
                    "type": "BalanceUpdated",
                    "data": {
                        "wallet_address": wallet_filter_value.clone(),
                        "old_balance": balance_qnk,
                        "new_balance": balance_qnk,
                        "change_reason": "SSE connection established",
                        "timestamp": chrono::Utc::now().to_rfc3339()
                    }
                });

                if let Ok(json) = serde_json::to_string(&initial_balance_event) {
                    // Return initial balance event, then continue with normal stream
                    // Set state_opt to None so we don't send initial balance again
                    return Some((
                        Ok(Event::default().event("balance-updated").data(json)),
                        (rx, filter, None, None, headers_only, miner_mode, connection_start, max_sse_lifetime),
                    ));
                }
            }

            // Normal SSE event loop (continues after initial balance sent)
            // v8.2.3: Use timeout on recv to allow Axum to detect dead connections.
            // Without timeout, rx.recv() blocks forever and CLOSE-WAIT sockets accumulate
            // because the stream never yields for Axum to attempt a keepalive write.
            loop {
                // v9.8.4: Enforce max SSE connection lifetime to prevent unbounded accumulation.
                // Miners auto-reconnect per SSE spec.
                if connection_start.elapsed() > max_sse_lifetime {
                    debug!("📡 SSE connection exceeded {}s lifetime, closing for reconnect", max_sse_lifetime.as_secs());
                    return None;
                }
                match tokio::time::timeout(
                    std::time::Duration::from_secs(30),
                    rx.recv()
                ).await {
                    Ok(Ok(event)) => {
                        // v9.0.1: miner_mode — only forward mining-relevant events.
                        // Drops ~80 KB/s of noise (MetricsUpdate, NodeStatusUpdate,
                        // TokenPriceUpdate, LiquidityPoolUpdate, emails, calendar, etc.)
                        if miner_mode {
                            match &event {
                                StreamEvent::NewBlock { .. }
                                | StreamEvent::MiningReward { .. }
                                | StreamEvent::BalanceUpdated { .. }
                                | StreamEvent::PendingMiningReward { .. }
                                | StreamEvent::MiningStats { .. }
                                | StreamEvent::MiningModeSwitch { .. } => {
                                    // These are mining-relevant — keep them
                                }
                                _ => {
                                    // Everything else is noise for miners — skip
                                    continue;
                                }
                            }
                        }

                        // Filter event based on wallet address
                        if !is_event_relevant(&event, &filter) {
                            // Skip this event, continue to next
                            continue;
                        }

                        // v1.0.2: Compact block events when headers_only=true (~100 bytes vs ~2-5KB)
                        let json_result = if headers_only {
                            match &event {
                                StreamEvent::NewBlock { height, hash, prev_hash, dag_round, timestamp, .. } => {
                                    serde_json::to_string(&serde_json::json!({
                                        "type": "NewBlock",
                                        "data": { "height": height, "hash": hash, "prev_hash": prev_hash, "dag_round": dag_round, "timestamp": timestamp }
                                    }))
                                }
                                StreamEvent::BlockFinalized { height, round, timestamp, .. } => {
                                    serde_json::to_string(&serde_json::json!({
                                        "type": "BlockFinalized",
                                        "data": { "height": height, "round": round, "timestamp": timestamp }
                                    }))
                                }
                                _ => serde_json::to_string(&event),
                            }
                        } else {
                            serde_json::to_string(&event)
                        };

                        match json_result {
                            Ok(json) => {
                                let event_name = event_type_name(&event);
                                // v3.4.0: Downgrade to debug! to reduce log spam
                                debug!(
                                    "📤 [SSE SEND] Sending {} event to wallet filter (first 16): {:?}",
                                    event_name,
                                    filter.as_ref().map(|f| &f[..16.min(f.len())])
                                );
                                return Some((
                                    Ok(Event::default().event(event_name).data(json)),
                                    (rx, filter, None, None, headers_only, miner_mode, connection_start, max_sse_lifetime),
                                ));
                            }
                            Err(e) => {
                                error!("Failed to serialize event: {}", e);
                                return Some((Err(axum::Error::new(e)), (rx, filter, None, None, headers_only, miner_mode, connection_start, max_sse_lifetime)));
                            }
                        }
                    }
                    Ok(Err(e)) => {
                        return match e {
                            tokio::sync::broadcast::error::RecvError::Lagged(n) => {
                                warn!("SSE client lagged behind by {} events, continuing", n);
                                Some((
                                    Ok(Event::default()
                                        .event("sse-lag")
                                        .data(format!("{{\"lagged_events\": {}}}", n))),
                                    (rx, filter, None, None, headers_only, miner_mode, connection_start, max_sse_lifetime),
                                ))
                            }
                            tokio::sync::broadcast::error::RecvError::Closed => {
                                debug!("SSE broadcast channel closed");
                                None // End the stream
                            }
                        };
                    }
                    Err(_timeout) => {
                        // v8.2.3: Timeout expired — yield a heartbeat event so Axum
                        // attempts to write to the socket. If the client is gone, the write
                        // will fail and Axum will drop this stream, cleaning up the CLOSE-WAIT.
                        return Some((
                            Ok(Event::default().comment("heartbeat")),
                            (rx, filter, None, None, headers_only, miner_mode, connection_start, max_sse_lifetime),
                        ));
                    }
                }
            }
        },
    );

    Ok(Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("keep-alive"),
    ))
}

/// v1.0.2: Delta-compressed WebSocket endpoint for bandwidth-efficient streaming
/// Usage: GET /ws/events?wallet_address=<address>&headers_only=true
/// Sends initial full state snapshot, then only delta updates
pub async fn websocket_delta_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
    Query(params): Query<WsQueryParams>,
) -> Response {
    info!("New WebSocket delta client connecting (headers_only={})", params.headers_only);
    ws.on_upgrade(move |socket| websocket_delta_connection(socket, state, params))
}

/// Handle delta-compressed WebSocket connection
async fn websocket_delta_connection(socket: WebSocket, state: Arc<AppState>, params: WsQueryParams) {
    let (mut sender, mut receiver) = FuturesStreamExt::split(socket);
    let mut rx = state.event_broadcaster.subscribe();
    let headers_only = params.headers_only;

    // Send initial state snapshot
    let status = state.node_status.read().await.clone();
    let initial = serde_json::json!({
        "type": "InitialState",
        "data": {
            "current_height": status.current_height,
            "connected_peers": status.connected_peers,
            "is_validator": status.is_validator,
        }
    });
    if let Ok(json) = serde_json::to_string(&initial) {
        if sender.send(Message::Text(json)).await.is_err() {
            return;
        }
    }

    // Spawn receiver task for client commands
    tokio::spawn(async move {
        while let Some(msg) = TokioStreamExt::next(&mut receiver).await {
            match msg {
                Ok(Message::Close(_)) | Err(_) => break,
                _ => {}
            }
        }
    });

    // Stream events with optional headers_only compaction
    loop {
        match tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv()).await {
            Ok(Ok(event)) => {
                let json_result = if headers_only {
                    match &event {
                        StreamEvent::NewBlock { height, hash, prev_hash, dag_round, timestamp, .. } => {
                            serde_json::to_string(&serde_json::json!({
                                "type": "NewBlock",
                                "data": { "height": height, "hash": hash, "prev_hash": prev_hash, "dag_round": dag_round, "timestamp": timestamp }
                            }))
                        }
                        StreamEvent::BlockFinalized { height, round, timestamp, .. } => {
                            serde_json::to_string(&serde_json::json!({
                                "type": "BlockFinalized",
                                "data": { "height": height, "round": round, "timestamp": timestamp }
                            }))
                        }
                        _ => serde_json::to_string(&event),
                    }
                } else {
                    serde_json::to_string(&event)
                };

                if let Ok(json) = json_result {
                    if sender.send(Message::Text(json)).await.is_err() {
                        break;
                    }
                }
            }
            Ok(Err(broadcast::error::RecvError::Lagged(n))) => {
                warn!("WebSocket delta client lagged by {} events", n);
            }
            Ok(Err(broadcast::error::RecvError::Closed)) => break,
            Err(_) => {
                // Heartbeat
                if sender.send(Message::Ping(vec![])).await.is_err() {
                    break;
                }
            }
        }
    }
    debug!("WebSocket delta client disconnected");
}

/// WebSocket endpoint for ultra-low latency streaming
/// Usage: GET /api/v1/ws
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> Response {
    info!("New WebSocket client attempting connection");

    ws.on_upgrade(|socket| websocket_connection(socket, state))
}

/// Handle individual WebSocket connection
async fn websocket_connection(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = FuturesStreamExt::split(socket);
    let mut rx = state.event_broadcaster.subscribe();

    info!("WebSocket client connected");

    // Send welcome message
    let welcome = StreamEvent::NodeStatusUpdate {
        status: state.node_status.read().await.clone(),
        timestamp: chrono::Utc::now(),
    };

    if let Ok(welcome_json) = serde_json::to_string(&welcome) {
        if sender.send(Message::Text(welcome_json)).await.is_err() {
            warn!("Failed to send welcome message to WebSocket client");
            return;
        }
    }

    // Spawn task to handle incoming messages from client
    let state_clone = state.clone();
    tokio::spawn(async move {
        while let Some(msg) = TokioStreamExt::next(&mut receiver).await {
            match msg {
                Ok(Message::Text(text)) => {
                    debug!("Received WebSocket message: {}", text);
                    // Handle client commands (subscription filters, etc.)
                    if let Err(e) = handle_client_message(&text, &state_clone).await {
                        warn!("Failed to handle client message: {}", e);
                    }
                }
                Ok(Message::Close(_)) => {
                    info!("WebSocket client disconnected");
                    break;
                }
                Ok(Message::Ping(_data)) => {
                    debug!("Received WebSocket ping");
                    // Pong is sent automatically by axum
                }
                Ok(_) => {
                    // Ignore other message types
                }
                Err(e) => {
                    warn!("WebSocket error: {}", e);
                    break;
                }
            }
        }
    });

    // Main event streaming loop - optimized for <50ms latency
    while let Ok(event) = rx.recv().await {
        let start_time = std::time::Instant::now();

        let event_json = match serde_json::to_string(&event) {
            Ok(json) => json,
            Err(e) => {
                error!("Failed to serialize event for WebSocket: {}", e);
                continue;
            }
        };

        if sender.send(Message::Text(event_json)).await.is_err() {
            info!("WebSocket client disconnected during send");
            break;
        }

        let latency = start_time.elapsed();
        if latency > std::time::Duration::from_millis(50) {
            warn!(
                "WebSocket event delivery took {}ms (target: <50ms)",
                latency.as_millis()
            );
        } else {
            debug!("WebSocket event delivered in {}ms", latency.as_millis());
        }
    }

    info!("WebSocket connection closed");
}

/// Handle messages from WebSocket clients
async fn handle_client_message(message: &str, _state: &Arc<AppState>) -> anyhow::Result<()> {
    // Parse client commands (future enhancement)
    #[derive(serde::Deserialize)]
    #[serde(tag = "command")]
    enum ClientCommand {
        Subscribe { event_types: Vec<String> },
        Unsubscribe { event_types: Vec<String> },
        GetStatus,
    }

    match serde_json::from_str::<ClientCommand>(message) {
        Ok(command) => {
            match command {
                ClientCommand::Subscribe { event_types } => {
                    debug!("Client subscribing to events: {:?}", event_types);
                    // TODO: Implement per-client filtering
                }
                ClientCommand::Unsubscribe { event_types } => {
                    debug!("Client unsubscribing from events: {:?}", event_types);
                    // TODO: Implement per-client filtering
                }
                ClientCommand::GetStatus => {
                    debug!("Client requesting status");
                    // TODO: Send current status
                }
            }
        }
        Err(_) => {
            debug!("Received non-JSON message from client: {}", message);
        }
    }

    Ok(())
}

/// Get event type name for SSE event naming
fn event_type_name(event: &StreamEvent) -> String {
    match event {
        StreamEvent::TransactionSubmitted { .. } => "transaction-submitted".to_string(),
        StreamEvent::TransactionStatusUpdate { .. } => "transaction-status".to_string(),
        StreamEvent::VertexCreated { .. } => "vertex-created".to_string(),
        StreamEvent::CertificateGenerated { .. } => "certificate-generated".to_string(),
        StreamEvent::BlockFinalized { .. } => "block-finalized".to_string(),
        StreamEvent::NewBlock { .. } => "new-block".to_string(),
        StreamEvent::NodeStatusUpdate { .. } => "node-status".to_string(),
        StreamEvent::PeerEvent { .. } => "peer-event".to_string(),
        StreamEvent::MetricsUpdate { .. } => "metrics-update".to_string(),
        StreamEvent::PeerDiscovered { .. } => "peer-discovered".to_string(),
        StreamEvent::PeerConnected { .. } => "peer-connected".to_string(),
        StreamEvent::PeerDisconnected { .. } => "peer-disconnected".to_string(),
        StreamEvent::PhantomPeerDiscovered { .. } => "phantom-peer-discovered".to_string(),
        StreamEvent::PhantomMessageReceived { .. } => "phantom-message-received".to_string(),
        StreamEvent::SecurityAlert { .. } => "security-alert".to_string(),
        StreamEvent::NetworkTopologyChanged { .. } => "network-topology-changed".to_string(),
        StreamEvent::TorCircuitEvent { .. } => "tor-circuit-event".to_string(),
        // v7.0.0: FaucetDispensed removed
        StreamEvent::BalanceUpdated { .. } => "balance-updated".to_string(),
        StreamEvent::TokenBalanceUpdated { .. } => "token-balance-updated".to_string(),
        // v6.0.2: Generic event names — "privacy-mixing-*" leaked that mixer was in use
        StreamEvent::PrivacyMixingStarted { .. } => "transaction-processing".to_string(),
        StreamEvent::PrivacyMixingCompleted { .. } => "transaction-confirmed".to_string(),
        StreamEvent::NitroBoost { .. } => "nitro_boost".to_string(),
        StreamEvent::NitroBoostsUpdate { .. } => "nitro_boosts_update".to_string(),
        StreamEvent::TokenPriceUpdate { .. } => "token_price_update".to_string(),
        StreamEvent::LiquidityPoolUpdate { .. } => "liquidity_pool_update".to_string(),
        StreamEvent::SwapExecuted { .. } => "swap_executed".to_string(),
        StreamEvent::MiningReward { .. } => "mining_reward".to_string(),
        StreamEvent::MiningStats { .. } => "mining_stats".to_string(),
        StreamEvent::PendingMiningReward { .. } => "pending_mining_reward".to_string(),
        StreamEvent::Custom { event_type, .. } => event_type.clone(),
        // v1.4.3: QNO events
        StreamEvent::QnoOracleUpdate { .. } => "oracle-update".to_string(),
        StreamEvent::QnoResolution { .. } => "qno-resolution".to_string(),
        StreamEvent::QnoStake { .. } => "qno-stake".to_string(),
        StreamEvent::QnoSlashing { .. } => "qno-slashing".to_string(),
        StreamEvent::ServerVersion { .. } => "server-version".to_string(),
        StreamEvent::StateSyncComplete { .. } => "state-sync-complete".to_string(),
        StreamEvent::PoolStatsUpdated { .. } => "pool-stats-updated".to_string(),
        StreamEvent::PoolShareAccepted { .. } => "pool-share-accepted".to_string(),
        StreamEvent::PoolBlockFound { .. } => "pool-block-found".to_string(),
        StreamEvent::PoolPayout { .. } => "pool-payout".to_string(),
        StreamEvent::EmailReceived { .. } => "email-received".to_string(),
        StreamEvent::EmailSent { .. } => "email-sent".to_string(),
        StreamEvent::EmailUnreadCount { .. } => "email-unread-count".to_string(),
        StreamEvent::CalendarEventCreated { .. } => "calendar-event-created".to_string(),
        StreamEvent::CalendarReminder { .. } => "calendar-reminder".to_string(),
        StreamEvent::ScheduledTransactionExecuted { .. } => "scheduled-tx-executed".to_string(),
        StreamEvent::MiningModeSwitch { .. } => "mining-mode-switch".to_string(),
        StreamEvent::ComputeStatus { .. } => "compute-status".to_string(),
        StreamEvent::GameTick { .. } => "game-tick".to_string(),
        StreamEvent::GameActionProcessed { .. } => "game-action-processed".to_string(),
        StreamEvent::GameStateSnapshot { .. } => "game-state-snapshot".to_string(),
        StreamEvent::GameEvent { .. } => "game-event".to_string(),
    }
}

/// High-performance event emitter with batching support
pub struct HighPerformanceEmitter {
    broadcaster: Arc<EventBroadcaster>,
    batch_size: usize,
    batch_timeout: std::time::Duration,
    pending_events: tokio::sync::Mutex<Vec<StreamEvent>>,
}

impl HighPerformanceEmitter {
    pub fn new(broadcaster: Arc<EventBroadcaster>) -> Self {
        Self {
            broadcaster,
            batch_size: 10, // Batch up to 10 events
            batch_timeout: std::time::Duration::from_millis(10), // Or timeout after 10ms
            pending_events: tokio::sync::Mutex::new(Vec::new()),
        }
    }

    /// Emit a single event immediately (for critical updates)
    pub async fn emit_immediate(
        &self,
        event: StreamEvent,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        debug!(
            "HighPerformanceEmitter: emitting event {}",
            event_type_name(&event)
        );
        let start = std::time::Instant::now();
        let result = self.broadcaster.broadcast(event).await;
        let latency = start.elapsed();

        if latency > std::time::Duration::from_millis(5) {
            warn!("High latency event emission: {}ms", latency.as_millis());
        }

        // Always return Ok() - event emission failures shouldn't break the application
        match result {
            Ok(_) => {
                debug!("HighPerformanceEmitter: event emission successful");
                Ok(())
            }
            Err(broadcast::error::SendError(_)) => {
                // Event could not be sent (usually means no active subscribers)
                // v3.4.2: Reduced to trace to prevent log spam
                trace!("HighPerformanceEmitter: Event emission skipped: no active subscribers");
                Ok(())
            }
        }
    }

    /// Add event to batch (for non-critical updates)
    pub async fn emit_batched(&self, event: StreamEvent) {
        let mut pending = self.pending_events.lock().await;
        pending.push(event);

        if pending.len() >= self.batch_size {
            self.flush_batch(&mut pending).await;
        }
    }

    /// Flush pending events
    async fn flush_batch(&self, events: &mut Vec<StreamEvent>) {
        if events.is_empty() {
            return;
        }

        let start = std::time::Instant::now();
        let count = events.len();

        for event in events.drain(..) {
            if let Err(broadcast::error::SendError(_)) = self.broadcaster.broadcast(event).await {
                debug!("Batched event emission skipped: no active subscribers");
            }
        }

        let latency = start.elapsed();
        debug!("Flushed {} events in {}ms", count, latency.as_millis());
    }

    /// Start background batch flush task
    pub fn start_batch_flusher(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        let emitter_clone = self.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(emitter_clone.batch_timeout);
            loop {
                interval.tick().await;
                let mut pending = emitter_clone.pending_events.lock().await;
                emitter_clone.flush_batch(&mut pending).await;
            }
        })
    }

    // ============================================================================
    // Network Event Emission Methods
    // ============================================================================

    /// Emit peer discovered event
    pub async fn emit_peer_discovered(
        &self,
        node_id: String,
        confidence: f64,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::PeerDiscovered {
            node_id,
            confidence,
            method: "bitcoin".to_string(),
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// Emit peer connected event
    pub async fn emit_peer_connected(
        &self,
        node_id: String,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::PeerConnected {
            node_id,
            connection_type: "bitcoin-tor".to_string(),
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// Emit peer disconnected event
    pub async fn emit_peer_disconnected(
        &self,
        node_id: String,
        reason: String,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::PeerDisconnected {
            node_id,
            reason,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// Emit DNS-Phantom peer discovered event
    pub async fn emit_phantom_peer_discovered(
        &self,
        node_id: String,
        discovery_method: String,
        confidence: f64,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::PhantomPeerDiscovered {
            node_id,
            discovery_method,
            confidence,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// Emit DNS-Phantom message received event
    pub async fn emit_phantom_message(
        &self,
        from: String,
        message_type: String,
        size_bytes: usize,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::PhantomMessageReceived {
            from,
            message_type,
            size_bytes,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// Emit security alert event
    pub async fn emit_security_alert(
        &self,
        alert_type: String,
        description: String,
        risk_level: f64,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::SecurityAlert {
            alert_type,
            description,
            risk_level,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// Emit network topology changed event
    pub async fn emit_network_topology_changed(
        &self,
        total_peers: u32,
        direct_peers: u32,
        phantom_peers: u32,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::NetworkTopologyChanged {
            total_peers,
            direct_peers,
            phantom_peers,
            timestamp: chrono::Utc::now(),
        };
        self.emit_batched(event).await;
        Ok(())
    }

    /// Emit Tor circuit event
    pub async fn emit_tor_circuit_event(
        &self,
        circuit_id: u32,
        event_type: String,
        details: String,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::TorCircuitEvent {
            circuit_id,
            event_type,
            details,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// Emit nitro boost event for a token
    pub async fn emit_nitro_boost(
        &self,
        token_id: String,
        points: u64,
        total_points: u64,
        boosted_by: String,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::NitroBoost {
            token_id,
            points,
            total_points,
            boosted_by,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// Emit bulk nitro boosts update
    pub async fn emit_nitro_boosts_update(
        &self,
        boosts: std::collections::HashMap<String, u64>,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::NitroBoostsUpdate {
            boosts,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// Emit token price update
    /// v2.9.22-beta: Added change_1h and change_7d parameters for full price metrics
    /// v2.9.25-beta: Added token_address parameter for frontend matching
    pub async fn emit_token_price_update(
        &self,
        token_symbol: String,
        token_address: Option<String>,  // 🆕 v2.9.25-beta
        price: f64,
        change_1h: f64,   // 🆕 v2.9.22-beta
        change_24h: f64,
        change_7d: f64,   // 🆕 v2.9.22-beta
        volume_24h: f64,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::TokenPriceUpdate {
            token_symbol,
            token_address,
            price,
            change_1h,
            change_24h,
            change_7d,
            volume_24h,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// Emit liquidity pool update
    pub async fn emit_liquidity_pool_update(
        &self,
        pool_id: String,
        token0: String,
        token1: String,
        reserve0: u64,
        reserve1: u64,
        total_liquidity: u64,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::LiquidityPoolUpdate {
            pool_id,
            token0,
            token1,
            reserve0,
            reserve1,
            total_liquidity,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// Emit swap executed event
    pub async fn emit_swap_executed(
        &self,
        from_token: String,
        to_token: String,
        amount_in: u128,
        amount_out: u128,
        wallet_address: String,
        price_impact: f64,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::SwapExecuted {
            from_token,
            to_token,
            amount_in,
            amount_out,
            wallet_address,
            price_impact,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// Emit mining reward event
    pub async fn emit_mining_reward(
        &self,
        miner_address: String,
        reward_qnk: f64,
        nonce: u64,
        block_height: u64,
        difficulty: String,
        hash_rate: f64,
        miner_id: Option<String>, // 🆕 v3.3.3-beta: Unique miner instance ID
        worker_name: Option<String>, // 🆕 v0.6.2-beta: Human-readable miner name
        origin_node_id: Option<String>, // 🆕 v2.3.5-beta: Which node mined this
        origin_node_name: Option<String>, // 🆕 v2.3.5-beta: Human-friendly node name
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::MiningReward {
            miner_address,
            reward_qnk,
            nonce,
            block_height,
            difficulty,
            hash_rate,
            miner_id,
            worker_name,
            origin_node_id,
            origin_node_name,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// Emit mining statistics update
    /// v3.2.25-beta: Added miner_id and worker_id parameters
    pub async fn emit_mining_stats(
        &self,
        miner_address: String,
        total_rewards: f64,
        total_blocks_found: u64,
        current_balance: f64,
        avg_hash_rate: f64,
        miner_id: Option<String>,
        worker_id: Option<String>,
        worker_name: Option<String>,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::MiningStats {
            miner_address,
            total_rewards,
            total_blocks_found,
            current_balance,
            avg_hash_rate,
            miner_id,
            worker_id,
            worker_name,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// v1.3.8-beta: Emit pending mining reward event
    /// This provides instant UI feedback for decentralized mining
    pub async fn emit_pending_mining_reward(
        &self,
        miner_address: String,
        pending_reward_qnk: f64,
        origin_node_id: String,
        source_height: u64,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::PendingMiningReward {
            miner_address,
            pending_reward_qnk,
            origin_node_id,
            source_height,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// v1.4.3: Emit QNO oracle update event
    pub async fn emit_qno_oracle_update(
        &self,
        domain: String,
        value: f64,
        confidence: f64,
        sources: Vec<OracleSourceInfo>,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::QnoOracleUpdate {
            domain,
            value,
            confidence,
            sources,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// v1.4.3: Emit QNO resolution result event
    pub async fn emit_qno_resolution(
        &self,
        stake_id: String,
        domain: String,
        predicted_value: f64,
        actual_value: f64,
        accuracy_score: f64,
        is_accurate: bool,
        slashing_applied: f64,
        reward_adjustment: f64,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::QnoResolution {
            stake_id,
            domain,
            predicted_value,
            actual_value,
            accuracy_score,
            is_accurate,
            slashing_applied,
            reward_adjustment,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// v1.4.3: Emit QNO stake placed event
    pub async fn emit_qno_stake(
        &self,
        stake_id: String,
        domain: String,
        amount: f64,
        confidence: f64,
        prediction_value: f64,
        wallet_address: String,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::QnoStake {
            stake_id,
            domain,
            amount,
            confidence,
            prediction_value,
            wallet_address,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// v1.4.3: Emit QNO slashing event
    pub async fn emit_qno_slashing(
        &self,
        stake_id: String,
        domain: String,
        amount_slashed: f64,
        reason: String,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::QnoSlashing {
            stake_id,
            domain,
            amount_slashed,
            reason,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// Emit pool stats updated event
    pub async fn emit_pool_stats_updated(
        &self,
        hashrate: f64,
        workers: usize,
        blocks_found: u64,
        current_round: u64,
        difficulty: f64,
        total_shares: u64,
        valid_shares: u64,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::PoolStatsUpdated {
            hashrate,
            workers,
            blocks_found,
            current_round,
            difficulty,
            total_shares,
            valid_shares,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// Emit pool share accepted event
    pub async fn emit_pool_share_accepted(
        &self,
        worker: String,
        difficulty: f64,
        share_id: String,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::PoolShareAccepted {
            worker,
            difficulty,
            share_id,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// Emit pool block found event
    pub async fn emit_pool_block_found(
        &self,
        block_hash: String,
        height: u64,
        finder: String,
        reward: f64,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::PoolBlockFound {
            block_hash,
            height,
            finder,
            reward,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }

    /// Emit pool payout event
    pub async fn emit_pool_payout(
        &self,
        batch_id: String,
        total_reward: f64,
        payout_count: usize,
        status: String,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::PoolPayout {
            batch_id,
            total_reward,
            payout_count,
            status,
            timestamp: chrono::Utc::now(),
        };
        self.emit_immediate(event).await
    }
}

/// Metrics for streaming performance
#[derive(Debug, Clone, serde::Serialize)]
pub struct StreamingMetrics {
    pub sse_connections: usize,
    pub websocket_connections: usize,
    pub total_events_sent: u64,
    pub avg_latency_ms: f64,
    pub max_latency_ms: u64,
    pub events_per_second: f64,
    pub buffer_utilization: f32,
}

impl StreamingMetrics {
    pub fn new() -> Self {
        Self {
            sse_connections: 0,
            websocket_connections: 0,
            total_events_sent: 0,
            avg_latency_ms: 0.0,
            max_latency_ms: 0,
            events_per_second: 0.0,
            buffer_utilization: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_event_broadcaster() {
        let broadcaster = EventBroadcaster::new();
        let _rx = broadcaster.subscribe();

        assert_eq!(broadcaster.subscriber_count(), 1);

        let event = StreamEvent::NodeStatusUpdate {
            status: NodeStatus {
                node_id: [1u8; 32],
                current_round: 1,
                current_height: 1,
                connected_peers: 0,
                tx_pool_size: 0,
                is_validator: true,
                uptime: std::time::Duration::from_secs(60),
            },
            timestamp: chrono::Utc::now(),
        };

        let result = broadcaster.broadcast(event).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_event_type_names() {
        let tx = Transaction {
            id: [1u8; 32],
            from: [2u8; 32],
            to: [3u8; 32],
            amount: 1000,
            fee: 10,
            nonce: 1,
            signature: vec![],
            timestamp: chrono::Utc::now(),
            data: vec![],
            token_type: q_types::TokenType::QUG,
            fee_token_type: q_types::TokenType::QUGUSD,
            tx_type: q_types::TransactionType::Transfer,
            pqc_signature: None,
            signature_phase: q_types::TxSignaturePhase::Phase0Ed25519,
            pqc_public_key: None,
        };

        let event = StreamEvent::TransactionSubmitted {
            transaction: tx,
            timestamp: chrono::Utc::now(),
        };

        assert_eq!(event_type_name(&event), "transaction-submitted");
    }

    #[tokio::test]
    async fn test_high_performance_emitter() {
        let broadcaster = Arc::new(EventBroadcaster::new());
        let emitter = HighPerformanceEmitter::new(broadcaster.clone());

        let event = StreamEvent::NodeStatusUpdate {
            status: NodeStatus {
                node_id: [1u8; 32],
                current_round: 1,
                current_height: 1,
                connected_peers: 0,
                tx_pool_size: 0,
                is_validator: true,
                uptime: std::time::Duration::from_secs(60),
            },
            timestamp: chrono::Utc::now(),
        };

        let result = emitter.emit_immediate(event).await;
        assert!(result.is_ok());
    }
}
