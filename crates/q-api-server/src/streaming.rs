/// Real-time streaming support for Q-NarwhalKnight
/// Provides both SSE (Server-Sent Events) and WebSocket streaming
/// Target latency: <50ms for critical updates
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::header,
    response::{
        sse::{Event, Sse},
        Response,
    },
};
use axum_extra::{headers, TypedHeader};
use futures_util::{sink::SinkExt, stream::StreamExt as FuturesStreamExt};
use q_types::*;
use serde_json;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio_stream::{wrappers::BroadcastStream, StreamExt as TokioStreamExt};
use tracing::{debug, error, info, warn};

use crate::AppState;

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
    /// Faucet tokens dispensed to wallet
    FaucetDispensed {
        wallet_address: String,
        amount_qnk: f64,
        balance_after: f64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Wallet balance updated (for transaction processing)
    BalanceUpdated {
        wallet_address: String,
        old_balance: f64,
        new_balance: f64,
        change_reason: String, // "transaction_sent", "transaction_received", "faucet", etc.
        timestamp: chrono::DateTime<chrono::Utc>,
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
    TokenPriceUpdate {
        token_symbol: String,
        price: f64,
        change_24h: f64,
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
        amount_in: u64,
        amount_out: u64,
        wallet_address: String,
        price_impact: f64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Custom event for mining rewards and other custom types
    Custom {
        event_type: String,
        data: serde_json::Value,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
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
}

impl EventBroadcaster {
    pub fn new() -> Self {
        let (tx, _rx) = broadcast::channel(10000); // High-capacity buffer
        Self { tx }
    }

    /// Broadcast an event to all subscribers
    pub fn broadcast(
        &self,
        event: StreamEvent,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let subscriber_count = self.tx.receiver_count();
        debug!(
            "Broadcasting event: {}, subscriber count: {}",
            event_type_name(&event),
            subscriber_count
        );

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
            // Silently succeed when no subscribers are present
            debug!("Event emission skipped: no active subscribers");
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

/// SSE endpoint for real-time event streaming
/// Usage: GET /api/v1/events
pub async fn sse_events(
    State(state): State<Arc<AppState>>,
    TypedHeader(user_agent): TypedHeader<headers::UserAgent>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, axum::Error>>> {
    info!("New SSE client connected: {}", user_agent.as_str());

    // Create a manual stream that keeps the receiver alive
    let rx = state.event_broadcaster.subscribe();
    let broadcaster_clone = state.event_broadcaster.clone();

    // Log initial subscriber count
    debug!(
        "SSE subscriber count after connection: {}",
        broadcaster_clone.subscriber_count()
    );

    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        match rx.recv().await {
            Ok(event) => match serde_json::to_string(&event) {
                Ok(json) => {
                    debug!("SSE sending event: {}", event_type_name(&event));
                    Some((
                        Ok(Event::default().event(event_type_name(&event)).data(json)),
                        rx,
                    ))
                }
                Err(e) => {
                    error!("Failed to serialize event: {}", e);
                    Some((Err(axum::Error::new(e)), rx))
                }
            },
            Err(e) => {
                match e {
                    tokio::sync::broadcast::error::RecvError::Lagged(n) => {
                        warn!("SSE client lagged behind by {} events, continuing", n);
                        Some((
                            Ok(Event::default()
                                .event("sse-lag")
                                .data(format!("{{\"lagged_events\": {}}}", n))),
                            rx,
                        ))
                    }
                    tokio::sync::broadcast::error::RecvError::Closed => {
                        debug!("SSE broadcast channel closed");
                        None // End the stream
                    }
                }
            }
        }
    });

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("keep-alive"),
    )
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
        StreamEvent::FaucetDispensed { .. } => "faucet-dispensed".to_string(),
        StreamEvent::BalanceUpdated { .. } => "balance-updated".to_string(),
        StreamEvent::PrivacyMixingStarted { .. } => "privacy-mixing-started".to_string(),
        StreamEvent::PrivacyMixingCompleted { .. } => "privacy-mixing-completed".to_string(),
        StreamEvent::NitroBoost { .. } => "nitro_boost".to_string(),
        StreamEvent::NitroBoostsUpdate { .. } => "nitro_boosts_update".to_string(),
        StreamEvent::TokenPriceUpdate { .. } => "token_price_update".to_string(),
        StreamEvent::LiquidityPoolUpdate { .. } => "liquidity_pool_update".to_string(),
        StreamEvent::SwapExecuted { .. } => "swap_executed".to_string(),
        StreamEvent::Custom { event_type, .. } => event_type.clone(),
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
        let result = self.broadcaster.broadcast(event);
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
                debug!("HighPerformanceEmitter: Event emission skipped: no active subscribers");
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
            if let Err(broadcast::error::SendError(_)) = self.broadcaster.broadcast(event) {
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
    pub async fn emit_token_price_update(
        &self,
        token_symbol: String,
        price: f64,
        change_24h: f64,
        volume_24h: f64,
    ) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let event = StreamEvent::TokenPriceUpdate {
            token_symbol,
            price,
            change_24h,
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
        amount_in: u64,
        amount_out: u64,
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

        let result = broadcaster.broadcast(event);
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
