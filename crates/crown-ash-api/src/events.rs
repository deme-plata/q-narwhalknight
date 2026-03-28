//! SSE event helpers for broadcasting Crown & Ash game events.
//!
//! These functions produce serialised event payloads suitable for emission
//! through the main `q-api-server` SSE stream.  They translate internal
//! [`GameEvent`] and [`TurnSummary`] values into JSON objects tagged with
//! an `event_type` field so the client can demultiplex the SSE stream.
//!
//! # Usage
//!
//! ```ignore
//! use crown_ash_api::events;
//!
//! let payload = events::turn_completed_payload(&summary);
//! // Feed `payload` into q-api-server's event_broadcaster as a Custom StreamEvent.
//! ```

use crown_ash_types::{GameEvent, TurnSummary};
use serde::Serialize;
use serde_json::Value;

// ─── SSE Event Type Constants ──────────────────────────────────────────────────

/// SSE event type emitted when a full game turn has been resolved.
pub const EVENT_TURN_COMPLETED: &str = "crown_ash_turn";

/// SSE event type emitted for individual notable game events (battles, deaths, etc.).
pub const EVENT_GAME_EVENT: &str = "crown_ash_event";

/// SSE event type emitted when a player joins the game.
pub const EVENT_PLAYER_JOINED: &str = "crown_ash_player_joined";

/// SSE event type emitted when the game world is initialized for the first time.
pub const EVENT_WORLD_INITIALIZED: &str = "crown_ash_world_init";

/// SSE event type emitted when the game world is reset.
pub const EVENT_WORLD_RESET: &str = "crown_ash_world_reset";

// ─── Payload Builders ──────────────────────────────────────────────────────────

/// Envelope wrapping all Crown & Ash SSE payloads.
#[derive(Debug, Serialize)]
struct SseEnvelope<T: Serialize> {
    event_type: &'static str,
    #[serde(flatten)]
    payload: T,
}

/// Build the JSON payload for a completed turn.
///
/// Contains the full [`TurnSummary`] with turn number, events, and aggregate
/// statistics.  Clients can use this to update their UI in one shot.
pub fn turn_completed_payload(summary: &TurnSummary) -> Value {
    let envelope = SseEnvelope {
        event_type: EVENT_TURN_COMPLETED,
        payload: summary,
    };
    serde_json::to_value(&envelope).unwrap_or_default()
}

/// Build the JSON payload for a single notable game event.
///
/// Emitted in real-time as events are generated during tick processing,
/// before the full turn summary is available.
pub fn game_event_payload(event: &GameEvent, turn: u32) -> Value {
    #[derive(Serialize)]
    struct EventWithTurn<'a> {
        turn: u32,
        event: &'a GameEvent,
    }

    let envelope = SseEnvelope {
        event_type: EVENT_GAME_EVENT,
        payload: EventWithTurn { turn, event },
    };
    serde_json::to_value(&envelope).unwrap_or_default()
}

/// Build the JSON payload for a player joining.
pub fn player_joined_payload(wallet: &str, faction_id: u8, faction_name: &str, turn: u32) -> Value {
    #[derive(Serialize)]
    struct PlayerJoinedData<'a> {
        wallet: &'a str,
        faction_id: u8,
        faction_name: &'a str,
        turn: u32,
    }

    let envelope = SseEnvelope {
        event_type: EVENT_PLAYER_JOINED,
        payload: PlayerJoinedData {
            wallet,
            faction_id,
            faction_name,
            turn,
        },
    };
    serde_json::to_value(&envelope).unwrap_or_default()
}

/// Build the JSON payload for world initialization.
pub fn world_initialized_payload(turn: u32, province_count: usize, faction_count: usize) -> Value {
    #[derive(Serialize)]
    struct WorldInitData {
        turn: u32,
        province_count: usize,
        faction_count: usize,
    }

    let envelope = SseEnvelope {
        event_type: EVENT_WORLD_INITIALIZED,
        payload: WorldInitData {
            turn,
            province_count,
            faction_count,
        },
    };
    serde_json::to_value(&envelope).unwrap_or_default()
}

/// Build the JSON payload for a world reset event.
pub fn world_reset_payload(reason: &str) -> Value {
    #[derive(Serialize)]
    struct WorldResetData<'a> {
        reason: &'a str,
    }

    let envelope = SseEnvelope {
        event_type: EVENT_WORLD_RESET,
        payload: WorldResetData { reason },
    };
    serde_json::to_value(&envelope).unwrap_or_default()
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crown_ash_types::TurnSummary;

    #[test]
    fn turn_payload_has_event_type() {
        let summary = TurnSummary {
            turn: 42,
            block_height: 1000,
            events: vec![],
            active_factions: 7,
            total_armies: 12,
            total_population: 250_000,
        };

        let val = turn_completed_payload(&summary);
        assert_eq!(val["event_type"], EVENT_TURN_COMPLETED);
        assert_eq!(val["turn"], 42);
        assert_eq!(val["active_factions"], 7);
    }

    #[test]
    fn player_joined_payload_structure() {
        let val = player_joined_payload("0xABC123", 3, "Salt League", 5);
        assert_eq!(val["event_type"], EVENT_PLAYER_JOINED);
        assert_eq!(val["wallet"], "0xABC123");
        assert_eq!(val["faction_id"], 3);
        assert_eq!(val["faction_name"], "Salt League");
        assert_eq!(val["turn"], 5);
    }

    #[test]
    fn world_init_payload_structure() {
        let val = world_initialized_payload(0, 25, 7);
        assert_eq!(val["event_type"], EVENT_WORLD_INITIALIZED);
        assert_eq!(val["province_count"], 25);
        assert_eq!(val["faction_count"], 7);
    }

    #[test]
    fn world_reset_payload_structure() {
        let val = world_reset_payload("admin reset");
        assert_eq!(val["event_type"], EVENT_WORLD_RESET);
        assert_eq!(val["reason"], "admin reset");
    }
}
