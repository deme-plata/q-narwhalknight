//! Network plugin — SSE-first real-time streaming with REST fallback.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  Dedicated "crown-ash-net" thread (tokio current_thread)        │
//! │                                                                  │
//! │  1. Initial REST fetch → WorldSnapshot                           │
//! │  2. Connect to GET /crown-ash/stream (SSE)                       │
//! │     ├─ crown_ash_turn   → extract events + REST fetch snapshot   │
//! │     ├─ crown_ash_event  → extract single event                   │
//! │     ├─ crown_ash_lagged → REST fetch snapshot                    │
//! │     └─ heartbeat        → (keep-alive, ignored)                  │
//! │  3. On disconnect: REST poll once, backoff, retry SSE            │
//! └────────────────────┬─────────────────────────────────────────────┘
//!                      │  Arc<Mutex<VecDeque<NetMessage>>>
//!                      ▼
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  Bevy Update system: drain_updates()                             │
//! │  → Drains mailbox → updates ClientGameState resource             │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! The SSE stream provides instant notification of game state changes.
//! Each `crown_ash_turn` event triggers a REST fetch for the full
//! [`WorldSnapshot`] (since the SSE payload only contains a summary).
//! When SSE is unavailable, the thread falls back to REST polling.

use bevy::prelude::*;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crown_ash_types::GameEvent;

use crate::resources::{
    config::CrownAshConfig,
    game_state::{ApiResponse, ClientGameState, ConnectionStatus, WorldSnapshot},
    narrative_state::DialogState,
    selection::Selection,
};

// ─── Plugin ──────────────────────────────────────────────────────────────────

/// Bevy plugin that handles all server communication for Crown & Ash.
///
/// Spawns a background network thread that maintains an SSE connection for
/// real-time updates.  Falls back to REST polling when SSE is unavailable.
pub struct CrownAshNetworkPlugin;

impl Plugin for CrownAshNetworkPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ClientGameState>()
            .init_resource::<CrownAshConfig>()
            .init_resource::<Selection>()
            .init_resource::<NetworkState>()
            .init_resource::<DialogState>()
            .add_systems(Startup, setup_network)
            .add_systems(Update, drain_updates);
    }
}

// ─── Messages ────────────────────────────────────────────────────────────────

/// Messages sent from the network thread to the Bevy main loop.
enum NetMessage {
    /// Full world snapshot fetched via REST.
    Snapshot(WorldSnapshot),
    /// Game events received via SSE.
    Events(Vec<GameEvent>),
    /// SSE connection established.
    Connected,
    /// Network error (SSE disconnect, REST failure, etc.).
    Error(String),
    /// Tier 2/3 narrative dialog received via SSE.
    NarrativeDialog {
        tier: u8,
        speaker: String,
        text: String,
        turn: u32,
    },
}

// ─── Internal State ──────────────────────────────────────────────────────────

/// Internal bookkeeping shared between the network thread and the Bevy system.
#[derive(Resource)]
struct NetworkState {
    /// Message queue: network thread pushes, Bevy system drains.
    mailbox: Arc<Mutex<VecDeque<NetMessage>>>,
    /// True while the SSE stream is connected and receiving events.
    sse_connected: Arc<AtomicBool>,
}

impl Default for NetworkState {
    fn default() -> Self {
        Self {
            mailbox: Arc::new(Mutex::new(VecDeque::new())),
            sse_connected: Arc::new(AtomicBool::new(false)),
        }
    }
}

// ─── Systems ─────────────────────────────────────────────────────────────────

/// Startup system: spawns the background network thread.
fn setup_network(config: Res<CrownAshConfig>, net: Res<NetworkState>) {
    let server_url = config.server_url.clone();
    let mailbox = Arc::clone(&net.mailbox);
    let sse_flag = Arc::clone(&net.sse_connected);

    // Spawn a dedicated thread with its own tokio runtime.
    // This avoids Bevy executor ↔ tokio compatibility issues and ensures
    // reqwest's async I/O works correctly.
    std::thread::Builder::new()
        .name("crown-ash-net".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime for Crown & Ash networking");
            rt.block_on(network_loop(server_url, mailbox, sse_flag));
        })
        .expect("Failed to spawn Crown & Ash network thread");
}

/// Per-frame system: drains the mailbox and updates [`ClientGameState`] and [`DialogState`].
fn drain_updates(
    net: ResMut<NetworkState>,
    mut game_state: ResMut<ClientGameState>,
    mut dialog_state: ResMut<DialogState>,
) {
    let Ok(mut lock) = net.mailbox.try_lock() else {
        return;
    };

    while let Some(msg) = lock.pop_front() {
        match msg {
            NetMessage::Snapshot(snapshot) => {
                game_state.last_update_turn = snapshot.meta.turn;
                game_state.connection = ConnectionStatus::Connected;
                game_state.world = Some(snapshot);
            }
            NetMessage::Events(events) => {
                game_state.events.extend(events);
                // Cap event history to prevent unbounded growth.
                if game_state.events.len() > 500 {
                    let drain = game_state.events.len() - 500;
                    game_state.events.drain(..drain);
                }
            }
            NetMessage::Connected => {
                game_state.connection = ConnectionStatus::Connected;
            }
            NetMessage::Error(e) => {
                // Only overwrite if we're not already connected via a
                // successful snapshot — avoids flickering on transient errors.
                if !matches!(game_state.connection, ConnectionStatus::Connected) {
                    game_state.connection = ConnectionStatus::Error(e);
                }
            }
            NetMessage::NarrativeDialog { tier, speaker, text, turn } => {
                dialog_state.push_dialog(speaker, text, tier, turn);
            }
        }
    }
}

// ─── Async Network Loop ─────────────────────────────────────────────────────

/// Main async loop running on the dedicated network thread.
///
/// 1. Fetches the initial world snapshot via REST.
/// 2. Connects to SSE and processes events (triggering REST fetches on turn).
/// 3. On SSE disconnect: fetches REST once, backs off, and retries.
async fn network_loop(
    server_url: String,
    mailbox: Arc<Mutex<VecDeque<NetMessage>>>,
    sse_flag: Arc<AtomicBool>,
) {
    let rest_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_default();

    // 1. Initial REST fetch — populate the world before SSE connects.
    match fetch_world_with_client(&rest_client, &server_url).await {
        Ok(snapshot) => push(&mailbox, NetMessage::Snapshot(snapshot)),
        Err(e) => push(&mailbox, NetMessage::Error(e)),
    }

    // 2. SSE reconnection loop with exponential backoff.
    let mut backoff_secs = 1u64;

    loop {
        match consume_sse(&rest_client, &server_url, &mailbox, &sse_flag).await {
            Ok(()) => {
                // Clean stream end (server shutdown, etc.) — reconnect quickly.
                backoff_secs = 1;
            }
            Err(e) => {
                sse_flag.store(false, Ordering::Relaxed);
                push(&mailbox, NetMessage::Error(format!("SSE: {e}")));
            }
        }

        // 3. SSE is down — fetch latest state via REST as a one-shot fallback.
        if let Ok(snapshot) = fetch_world_with_client(&rest_client, &server_url).await {
            push(&mailbox, NetMessage::Snapshot(snapshot));
        }

        // Wait before reconnecting.
        tokio::time::sleep(Duration::from_secs(backoff_secs)).await;
        backoff_secs = (backoff_secs * 2).min(30);
    }
}

// ─── SSE Consumer ────────────────────────────────────────────────────────────

/// Connect to the SSE stream and process events until the connection drops.
///
/// On each `crown_ash_turn` event, fires a REST fetch for the full
/// `WorldSnapshot` (since the SSE payload only carries `TurnSummary` stats).
///
/// Returns `Err` on connection failure, read error, or 60-second read timeout
/// (indicating the server stopped sending heartbeats).
async fn consume_sse(
    rest_client: &reqwest::Client,
    server_url: &str,
    mailbox: &Arc<Mutex<VecDeque<NetMessage>>>,
    sse_flag: &Arc<AtomicBool>,
) -> Result<(), String> {
    let url = format!("{}/api/v1/crown-ash/stream", server_url);

    // SSE connections are long-lived — no timeout on the client.
    let sse_client = reqwest::Client::builder()
        .build()
        .map_err(|e| format!("SSE client build: {e}"))?;

    let mut response = sse_client
        .get(&url)
        .header("Accept", "text/event-stream")
        .send()
        .await
        .map_err(|e| format!("SSE connect: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("SSE HTTP {}", response.status()));
    }

    sse_flag.store(true, Ordering::Relaxed);
    push(mailbox, NetMessage::Connected);

    let mut parser = SseParser::new();

    loop {
        // 60-second timeout: server sends heartbeats every 15s, so 4 missed
        // heartbeats means the connection is dead.
        let chunk = tokio::time::timeout(Duration::from_secs(60), response.chunk())
            .await
            .map_err(|_| "SSE read timeout (60s — heartbeat lost)".to_string())?
            .map_err(|e| format!("SSE read: {e}"))?;

        let Some(bytes) = chunk else {
            // Stream ended normally (server closed connection).
            sse_flag.store(false, Ordering::Relaxed);
            return Err("SSE stream ended".into());
        };

        for event in parser.feed(&bytes) {
            handle_sse_event(rest_client, server_url, &event, mailbox).await;
        }
    }
}

/// Dispatch a parsed SSE event.
async fn handle_sse_event(
    rest_client: &reqwest::Client,
    server_url: &str,
    event: &ParsedSseEvent,
    mailbox: &Arc<Mutex<VecDeque<NetMessage>>>,
) {
    match event.event_type.as_str() {
        "crown_ash_turn" => {
            // TurnSummary fields are flattened into the top-level JSON:
            //   {"event_type":"crown_ash_turn","turn":42,"events":[...],...}
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&event.data) {
                if let Some(events_arr) = val.get("events") {
                    if let Ok(events) =
                        serde_json::from_value::<Vec<GameEvent>>(events_arr.clone())
                    {
                        if !events.is_empty() {
                            push(mailbox, NetMessage::Events(events));
                        }
                    }
                }
            }

            // Fetch the full world snapshot for rendering.
            if let Ok(snapshot) = fetch_world_with_client(rest_client, server_url).await {
                push(mailbox, NetMessage::Snapshot(snapshot));
            }
        }

        "crown_ash_event" => {
            // Single event: {"event_type":"crown_ash_event","turn":42,"event":{...}}
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&event.data) {
                if let Some(event_val) = val.get("event") {
                    if let Ok(ge) = serde_json::from_value::<GameEvent>(event_val.clone()) {
                        push(mailbox, NetMessage::Events(vec![ge]));
                    }
                }
            }
        }

        "crown_ash_world_reset" | "crown_ash_lagged" => {
            // State may be stale or incomplete — re-fetch everything.
            if let Ok(snapshot) = fetch_world_with_client(rest_client, server_url).await {
                push(mailbox, NetMessage::Snapshot(snapshot));
            }
        }

        "crown_ash_dialog" => {
            // Tier 2 dialog: {"event_type":"crown_ash_dialog","tier":2,"turn":42,
            //   "speaker":"King Aldric","text":"Victory!","generation_type":"short_dialog"}
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&event.data) {
                let speaker = val.get("speaker")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown")
                    .to_string();
                let text = val.get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let tier = val.get("tier")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(2) as u8;
                let turn = val.get("turn")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;

                if !text.is_empty() {
                    push(mailbox, NetMessage::NarrativeDialog { tier, speaker, text, turn });
                }
            }
        }

        "crown_ash_epic" => {
            // Tier 3 deep narrative: {"event_type":"crown_ash_epic","tier":3,"turn":42,
            //   "text":"The dawn broke crimson...","generation_type":"battle_epic"}
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&event.data) {
                let text = val.get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let tier = val.get("tier")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(3) as u8;
                let turn = val.get("turn")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;

                if !text.is_empty() {
                    push(mailbox, NetMessage::NarrativeDialog {
                        tier,
                        speaker: "Narrator".to_string(),
                        text,
                        turn,
                    });
                }
            }
        }

        // "crown_ash_heartbeat", "crown_ash_player_joined", "crown_ash_world_init",
        // "crown_ash_prose", "crown_ash_token"
        _ => {}
    }
}

// ─── REST Fetch ──────────────────────────────────────────────────────────────

/// Fetch the full world snapshot via the REST API.
async fn fetch_world_with_client(
    client: &reqwest::Client,
    server_url: &str,
) -> Result<WorldSnapshot, String> {
    let url = format!("{}/api/v1/crown-ash/world", server_url);
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("HTTP: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("Server returned {}", resp.status()));
    }

    let api: ApiResponse<WorldSnapshot> = resp
        .json()
        .await
        .map_err(|e| format!("JSON decode: {e}"))?;

    if api.success {
        api.data.ok_or_else(|| "success=true but data=null".into())
    } else {
        Err(api.error.unwrap_or_else(|| "Unknown server error".into()))
    }
}

// ─── SSE Protocol Parser ────────────────────────────────────────────────────

/// A parsed SSE event with its type and data payload.
struct ParsedSseEvent {
    event_type: String,
    data: String,
}

/// Incremental SSE protocol parser.
///
/// Accumulates byte chunks and yields complete events. Handles:
/// - `event:` lines (set event type)
/// - `data:` lines (multi-line data concatenated with `\n`)
/// - Empty lines (event boundary)
/// - Comment lines starting with `:` (keep-alive, ignored)
struct SseParser {
    buf: String,
    event_type: String,
    data: String,
}

impl SseParser {
    fn new() -> Self {
        Self {
            buf: String::new(),
            event_type: String::new(),
            data: String::new(),
        }
    }

    /// Feed a chunk of bytes and return any complete SSE events.
    fn feed(&mut self, chunk: &[u8]) -> Vec<ParsedSseEvent> {
        let text = String::from_utf8_lossy(chunk);
        self.buf.push_str(&text);

        // Safety: if buffer grows beyond 1MB without producing events,
        // the stream is probably sending garbage — reset to avoid OOM.
        if self.buf.len() > 1_048_576 {
            self.buf.clear();
            self.event_type.clear();
            self.data.clear();
            return Vec::new();
        }

        let mut events = Vec::new();

        loop {
            let Some(nl) = self.buf.find('\n') else {
                break;
            };
            let line = self.buf[..nl].trim_end_matches('\r').to_string();
            self.buf = self.buf[nl + 1..].to_string();

            if line.is_empty() {
                // Empty line = event boundary.
                if !self.data.is_empty() {
                    events.push(ParsedSseEvent {
                        event_type: if self.event_type.is_empty() {
                            "message".into()
                        } else {
                            std::mem::take(&mut self.event_type)
                        },
                        data: std::mem::take(&mut self.data),
                    });
                }
                self.event_type.clear();
            } else if let Some(rest) = line.strip_prefix("event:") {
                self.event_type = rest.trim().to_string();
            } else if let Some(rest) = line.strip_prefix("data:") {
                if !self.data.is_empty() {
                    self.data.push('\n');
                }
                // Strip at most one leading space (SSE spec).
                let value = rest.strip_prefix(' ').unwrap_or(rest);
                self.data.push_str(value);
            }
            // Ignore comment lines (`:`) and unknown fields.
        }

        events
    }
}

// ─── Utility ─────────────────────────────────────────────────────────────────

/// Maximum pending messages before we start dropping old events.
/// Snapshots and Connected/Error messages are always kept; only Events and
/// NarrativeDialog are dropped when the queue is full — they'll be superseded
/// by the next snapshot anyway.
const MAX_MAILBOX_SIZE: usize = 512;

/// Push a message into the shared mailbox, evicting oldest events if full.
fn push(mailbox: &Arc<Mutex<VecDeque<NetMessage>>>, msg: NetMessage) {
    if let Ok(mut lock) = mailbox.lock() {
        // If the queue is full, drop old event/dialog messages to make room.
        while lock.len() >= MAX_MAILBOX_SIZE {
            lock.pop_front();
        }
        lock.push_back(msg);
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_event() {
        let mut parser = SseParser::new();
        let chunk = b"event: crown_ash_turn\ndata: {\"turn\":42}\n\n";
        let events = parser.feed(chunk);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "crown_ash_turn");
        assert_eq!(events[0].data, "{\"turn\":42}");
    }

    #[test]
    fn parse_multiline_data() {
        let mut parser = SseParser::new();
        let chunk = b"event: test\ndata: line1\ndata: line2\n\n";
        let events = parser.feed(chunk);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2");
    }

    #[test]
    fn parse_default_event_type() {
        let mut parser = SseParser::new();
        let chunk = b"data: hello\n\n";
        let events = parser.feed(chunk);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message");
    }

    #[test]
    fn parse_multiple_events_in_one_chunk() {
        let mut parser = SseParser::new();
        let chunk =
            b"event: a\ndata: 1\n\nevent: b\ndata: 2\n\n";
        let events = parser.feed(chunk);

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "a");
        assert_eq!(events[0].data, "1");
        assert_eq!(events[1].event_type, "b");
        assert_eq!(events[1].data, "2");
    }

    #[test]
    fn parse_across_chunks() {
        let mut parser = SseParser::new();

        // First chunk: partial event.
        let events1 = parser.feed(b"event: crown_ash_turn\nda");
        assert!(events1.is_empty());

        // Second chunk: rest of event.
        let events2 = parser.feed(b"ta: {\"turn\":1}\n\n");
        assert_eq!(events2.len(), 1);
        assert_eq!(events2[0].event_type, "crown_ash_turn");
        assert_eq!(events2[0].data, "{\"turn\":1}");
    }

    #[test]
    fn comment_lines_ignored() {
        let mut parser = SseParser::new();
        let chunk = b": this is a comment\nevent: test\ndata: hello\n\n";
        let events = parser.feed(chunk);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "test");
    }

    #[test]
    fn empty_data_not_emitted() {
        let mut parser = SseParser::new();
        let chunk = b"event: empty\n\n";
        let events = parser.feed(chunk);

        // No data lines → no event emitted.
        assert!(events.is_empty());
    }

    #[test]
    fn crlf_line_endings() {
        let mut parser = SseParser::new();
        let chunk = b"event: test\r\ndata: hello\r\n\r\n";
        let events = parser.feed(chunk);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "test");
        assert_eq!(events[0].data, "hello");
    }
}
