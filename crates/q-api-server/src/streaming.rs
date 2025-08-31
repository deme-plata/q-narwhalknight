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
use futures_util::{sink::SinkExt, stream::StreamExt};
use q_types::*;
use serde_json;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio_stream::{wrappers::BroadcastStream, StreamExt as _};
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
    pub fn broadcast(&self, event: StreamEvent) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        self.tx.send(event)
    }

    /// Subscribe to events
    pub fn subscribe(&self) -> broadcast::Receiver<StreamEvent> {
        self.tx.subscribe()
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

    let rx = state.event_broadcaster.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|result| match result {
        Ok(event) => {
            match serde_json::to_string(&event) {
                Ok(json) => {
                    let sse_event = Event::default()
                        .event(event_type_name(&event))
                        .data(json);
                    Some(Ok(sse_event))
                }
                Err(e) => {
                    error!("Failed to serialize event: {}", e);
                    None
                }
            }
        }
        Err(e) => {
            warn!("SSE stream error: {}", e);
            None
        }
    });

    Sse::new(stream)
        .keep_alive(
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
    let (mut sender, mut receiver) = socket.split();
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
        while let Some(msg) = receiver.next().await {
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
                Ok(Message::Ping(data)) => {
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
            warn!("WebSocket event delivery took {}ms (target: <50ms)", latency.as_millis());
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
fn event_type_name(event: &StreamEvent) -> &'static str {
    match event {
        StreamEvent::TransactionSubmitted { .. } => "transaction-submitted",
        StreamEvent::TransactionStatusUpdate { .. } => "transaction-status",
        StreamEvent::VertexCreated { .. } => "vertex-created",
        StreamEvent::CertificateGenerated { .. } => "certificate-generated",
        StreamEvent::BlockFinalized { .. } => "block-finalized",
        StreamEvent::NodeStatusUpdate { .. } => "node-status",
        StreamEvent::PeerEvent { .. } => "peer-event",
        StreamEvent::MetricsUpdate { .. } => "metrics-update",
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
    pub async fn emit_immediate(&self, event: StreamEvent) -> Result<(), broadcast::error::SendError<StreamEvent>> {
        let start = std::time::Instant::now();
        let result = self.broadcaster.broadcast(event);
        let latency = start.elapsed();
        
        if latency > std::time::Duration::from_millis(5) {
            warn!("High latency event emission: {}ms", latency.as_millis());
        }
        
        result
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
            if let Err(e) = self.broadcaster.broadcast(event) {
                warn!("Failed to broadcast batched event: {}", e);
            }
        }

        let latency = start.elapsed();
        debug!("Flushed {} events in {}ms", count, latency.as_millis());
    }

    /// Start background batch flush task
    pub fn start_batch_flusher(&self) -> tokio::task::JoinHandle<()> {
        let emitter = Arc::new(self);
        let emitter_clone = emitter.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(emitter_clone.batch_timeout);
            loop {
                interval.tick().await;
                let mut pending = emitter_clone.pending_events.lock().await;
                emitter_clone.flush_batch(&mut pending).await;
            }
        })
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