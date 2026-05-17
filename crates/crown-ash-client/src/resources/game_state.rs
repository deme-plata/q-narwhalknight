//! Client-side game state resource — single source of truth for all rendering.
//!
//! Defines local DTOs that match the JSON shape returned by the Crown & Ash
//! REST API (`GET /crown-ash/world`).  We re-define the response envelope here
//! so the client crate never depends on `crown-ash-api` (which pulls in axum,
//! q-storage, and the full server dependency tree).

use bevy::prelude::*;
use crown_ash_types::{
    Army, Character, DiplomaticRelation, Faction, GameEvent, Province, Realm, WorldMeta,
};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// API response DTOs
// ---------------------------------------------------------------------------

/// Generic API response envelope matching `q-api-server`'s `ApiResponse<T>`.
#[derive(Debug, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

/// Full world snapshot matching `GET /crown-ash/world` response body.
///
/// Every field maps 1:1 to a top-level key inside the `data` object.  All
/// inner types come from `crown_ash_types` which already derives
/// `Serialize + Deserialize`.
#[derive(Debug, Clone, Deserialize)]
pub struct WorldSnapshot {
    pub meta: WorldMeta,
    pub provinces: Vec<Province>,
    pub factions: Vec<Faction>,
    pub realms: Vec<Realm>,
    pub armies: Vec<Army>,
    pub characters: Vec<Character>,
    pub diplomacy: Vec<DiplomaticRelation>,
    pub action_queue_size: usize,
}

// ---------------------------------------------------------------------------
// Connection status
// ---------------------------------------------------------------------------

/// Connection status for the network plugin's polling loop.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ConnectionStatus {
    Disconnected,
    Connecting,
    Connected,
    Error(String),
}

// ---------------------------------------------------------------------------
// Main game-state resource
// ---------------------------------------------------------------------------

/// Central game-state resource inserted as a Bevy `Resource`.
///
/// Rendering systems read `world` to draw the map, UI panels, and overlays.
/// The network plugin is the **only** writer.
#[derive(Resource)]
pub struct ClientGameState {
    /// Latest world snapshot received from the server (None until first fetch).
    pub world: Option<WorldSnapshot>,
    /// Narrative events accumulated since the client started.
    pub events: Vec<GameEvent>,
    /// Current connection status (drives the UI status indicator).
    pub connection: ConnectionStatus,
    /// Turn number of the most recent successful fetch.
    pub last_update_turn: u32,
}

impl Default for ClientGameState {
    fn default() -> Self {
        Self {
            world: None,
            events: Vec::new(),
            connection: ConnectionStatus::Disconnected,
            last_update_turn: 0,
        }
    }
}
