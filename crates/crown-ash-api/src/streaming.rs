//! SSE streaming endpoint for Crown & Ash real-time game updates.
//!
//! Provides a `GET /stream` endpoint that clients can connect to for
//! server-pushed game events instead of polling `/world`.
//!
//! # Architecture
//!
//! A [`tokio::sync::broadcast`] channel carries [`StreamEvent`] values.
//! The simulation tick task sends events after each turn completes.
//! Each SSE client gets an independent receiver that replays from the
//! point of connection.
//!
//! # Event Types
//!
//! | SSE `event:` field        | Payload                         |
//! |---------------------------|---------------------------------|
//! | `crown_ash_turn`          | Full `TurnSummary`              |
//! | `crown_ash_event`         | Single `GameEvent` + turn       |
//! | `crown_ash_player_joined` | Wallet + faction info           |
//! | `crown_ash_world_init`    | Province/faction counts         |
//! | `crown_ash_world_reset`   | Reason string                   |
//! | `crown_ash_heartbeat`     | Turn number + timestamp         |

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use serde::Serialize;
use serde_json::Value;
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use tracing::debug;

// ─── Broadcast Infrastructure ────────────────────────────────────────────────

/// A single event on the Crown & Ash SSE stream.
#[derive(Debug, Clone, Serialize)]
pub struct StreamEvent {
    /// SSE `event:` field (e.g. "crown_ash_turn").
    pub event_type: String,
    /// JSON payload.
    pub data: Value,
}

/// Sender half of the broadcast channel.
///
/// The simulation tick task calls `sender.send(event)` after each turn.
/// Clone the `Arc` to share with multiple producers.
pub type EventSender = Arc<broadcast::Sender<StreamEvent>>;

/// Create a new broadcast channel for Crown & Ash SSE events.
///
/// Returns the sender (for the tick loop) and a receiver factory.
/// `capacity` is the number of buffered events before lagging receivers
/// start dropping messages (they'll get a `Lagged` error and resume).
pub fn create_event_channel(capacity: usize) -> EventSender {
    let (tx, _rx) = broadcast::channel(capacity);
    Arc::new(tx)
}

// ─── SSE Handler ─────────────────────────────────────────────────────────────

/// SSE state passed to the streaming handler via Axum's `State` extractor.
#[derive(Clone)]
pub struct SseState {
    pub sender: EventSender,
}

/// `GET /stream` — Server-Sent Events endpoint for real-time game updates.
///
/// Each connection gets its own broadcast receiver. Events that arrive
/// before the client connects are not replayed (connect early!).
///
/// The stream sends a heartbeat every 15 seconds to keep proxies alive.
pub async fn sse_stream(
    State(sse_state): State<SseState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = sse_state.sender.subscribe();

    let stream = BroadcastStream::new(rx)
        .filter_map(|result| {
            match result {
                Ok(event) => {
                    match Event::default()
                        .event(&event.event_type)
                        .json_data(&event.data)
                    {
                        Ok(sse_event) => Some(Ok(sse_event)),
                        Err(_) => None, // skip malformed events
                    }
                }
                Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(n)) => {
                    debug!("Crown & Ash SSE client lagged by {} events", n);
                    // Send a warning event so the client knows it missed messages.
                    Some(Ok(
                        Event::default()
                            .event("crown_ash_lagged")
                            .data(format!("{{\"missed_events\":{}}}", n))
                    ))
                }
            }
        });

    Sse::new(stream)
        .keep_alive(KeepAlive::default().interval(std::time::Duration::from_secs(15)))
}

// ─── Helper: Send Events ─────────────────────────────────────────────────────

/// Broadcast a pre-built SSE event to all connected clients.
///
/// Returns the number of active receivers, or 0 if nobody is listening.
pub fn broadcast_event(sender: &EventSender, event_type: &str, data: Value) -> usize {
    let event = StreamEvent {
        event_type: event_type.to_string(),
        data,
    };
    sender.send(event).unwrap_or(0)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_creation_works() {
        let sender = create_event_channel(64);
        assert_eq!(sender.receiver_count(), 0);
    }

    #[test]
    fn broadcast_with_no_receivers_returns_zero() {
        let sender = create_event_channel(64);
        let count = broadcast_event(&sender, "test", serde_json::json!({"hello": "world"}));
        assert_eq!(count, 0);
    }

    #[test]
    fn broadcast_with_receiver_delivers() {
        let sender = create_event_channel(64);
        let mut rx = sender.subscribe();

        let count = broadcast_event(&sender, "crown_ash_turn", serde_json::json!({"turn": 42}));
        assert_eq!(count, 1);

        let event = rx.try_recv().expect("Should have received event");
        assert_eq!(event.event_type, "crown_ash_turn");
        assert_eq!(event.data["turn"], 42);
    }
}
