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

/// SSE event type for Tier 1 narrative prose (template-generated, instant).
pub const EVENT_NARRATIVE_PROSE: &str = "crown_ash_prose";

/// SSE event type for Tier 2 short dialog (LLM-generated, 1-3s).
pub const EVENT_NARRATIVE_DIALOG: &str = "crown_ash_dialog";

/// SSE event type for Tier 3 deep narrative (LLM-generated, 5-15s).
pub const EVENT_NARRATIVE_EPIC: &str = "crown_ash_epic";

/// SSE event type for streaming tokens (progressive LLM display).
pub const EVENT_NARRATIVE_TOKEN: &str = "crown_ash_token";

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

/// Build the JSON payload for a Tier 1 narrative prose event.
///
/// Contains the template-generated prose for a single event, color-coded by
/// importance. Sent immediately after the event occurs (zero latency).
pub fn narrative_prose_payload(
    turn: u32,
    prose: &str,
    summary: &str,
    importance: &crown_ash_narrative::Importance,
) -> Value {
    #[derive(Serialize)]
    struct NarrativeProseData<'a> {
        tier: u8,
        turn: u32,
        prose: &'a str,
        summary: &'a str,
        importance: &'a str,
    }

    let importance_str = match importance {
        crown_ash_narrative::Importance::Minor => "minor",
        crown_ash_narrative::Importance::Notable => "notable",
        crown_ash_narrative::Importance::Epic => "epic",
    };

    let envelope = SseEnvelope {
        event_type: EVENT_NARRATIVE_PROSE,
        payload: NarrativeProseData {
            tier: 1,
            turn,
            prose,
            summary,
            importance: importance_str,
        },
    };
    serde_json::to_value(&envelope).unwrap_or_default()
}

/// Build the JSON payload for a Tier 2 dialog event (LLM-generated).
pub fn narrative_dialog_payload(
    turn: u32,
    speaker: &str,
    text: &str,
    generation_type: &str,
) -> Value {
    #[derive(Serialize)]
    struct DialogData<'a> {
        tier: u8,
        turn: u32,
        speaker: &'a str,
        text: &'a str,
        generation_type: &'a str,
    }

    let envelope = SseEnvelope {
        event_type: EVENT_NARRATIVE_DIALOG,
        payload: DialogData { tier: 2, turn, speaker, text, generation_type },
    };
    serde_json::to_value(&envelope).unwrap_or_default()
}

/// Build the JSON payload for a Tier 3 deep narrative event (LLM-generated).
pub fn narrative_epic_payload(
    turn: u32,
    text: &str,
    generation_type: &str,
) -> Value {
    #[derive(Serialize)]
    struct EpicData<'a> {
        tier: u8,
        turn: u32,
        text: &'a str,
        generation_type: &'a str,
    }

    let envelope = SseEnvelope {
        event_type: EVENT_NARRATIVE_EPIC,
        payload: EpicData { tier: 3, turn, text, generation_type },
    };
    serde_json::to_value(&envelope).unwrap_or_default()
}

/// Build the JSON payload for a streaming token (Tier 2/3 progressive display).
pub fn narrative_token_payload(tier: u8, token: &str, generation_type: &str) -> Value {
    #[derive(Serialize)]
    struct TokenData<'a> {
        tier: u8,
        token: &'a str,
        generation_type: &'a str,
    }

    let envelope = SseEnvelope {
        event_type: EVENT_NARRATIVE_TOKEN,
        payload: TokenData { tier, token, generation_type },
    };
    serde_json::to_value(&envelope).unwrap_or_default()
}

/// Process all events for a turn through the cascade engine and broadcast
/// Tier 0 + Tier 1 results immediately via SSE.
///
/// Returns the cascade results for Tier 2/3 deferred LLM processing.
pub fn broadcast_cascade_narratives(
    sender: &super::streaming::EventSender,
    events: &[GameEvent],
    turn: u32,
    ctx: &crown_ash_narrative::WorldContext,
) -> crown_ash_narrative::cascade::TurnCascade {
    let cascade = crown_ash_narrative::cascade::CascadeEngine::new();
    let turn_cascade = cascade.process_turn(turn, events, ctx);

    // Broadcast Tier 1 prose for each event immediately
    for result in &turn_cascade.results {
        let payload = narrative_prose_payload(
            turn,
            &result.prose,
            &result.summary,
            &result.importance,
        );
        let _ = super::streaming::broadcast_event(
            sender,
            EVENT_NARRATIVE_PROSE,
            payload,
        );
    }

    turn_cascade
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

    #[test]
    fn narrative_prose_payload_structure() {
        let val = narrative_prose_payload(
            42,
            "Steel met steel on the plains.",
            "Battle at Ashenmere",
            &crown_ash_narrative::Importance::Epic,
        );
        assert_eq!(val["event_type"], EVENT_NARRATIVE_PROSE);
        assert_eq!(val["tier"], 1);
        assert_eq!(val["turn"], 42);
        assert_eq!(val["importance"], "epic");
        assert!(val["prose"].as_str().unwrap().contains("Steel"));
    }

    #[test]
    fn narrative_dialog_payload_structure() {
        let val = narrative_dialog_payload(42, "King Aldric", "Victory is ours!", "short_dialog");
        assert_eq!(val["event_type"], EVENT_NARRATIVE_DIALOG);
        assert_eq!(val["tier"], 2);
        assert_eq!(val["speaker"], "King Aldric");
        assert!(val["text"].as_str().unwrap().contains("Victory"));
    }

    #[test]
    fn narrative_epic_payload_structure() {
        let val = narrative_epic_payload(42, "The dawn broke crimson over the battlefield.", "battle_epic");
        assert_eq!(val["event_type"], EVENT_NARRATIVE_EPIC);
        assert_eq!(val["tier"], 3);
        assert!(val["text"].as_str().unwrap().contains("dawn"));
    }

    #[test]
    fn narrative_token_payload_structure() {
        let val = narrative_token_payload(2, "The", "short_dialog");
        assert_eq!(val["event_type"], EVENT_NARRATIVE_TOKEN);
        assert_eq!(val["tier"], 2);
        assert_eq!(val["token"], "The");
    }
}
