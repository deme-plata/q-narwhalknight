//! Crown & Ash — REST API handlers for the medieval grand strategy game.
//!
//! This crate provides Axum route handlers that expose game state over HTTP.
//! It is designed to be mounted into the main `q-api-server` router under
//! a `/api/v1/crown-ash` prefix.
//!
//! # Architecture
//!
//! To avoid circular dependencies with `q-api-server`, the game state is held
//! in [`CrownAshGameState`] behind an `Arc<tokio::sync::RwLock<..>>`.  The
//! main server constructs this state, optionally loads a persisted snapshot,
//! and passes it to [`create_crown_ash_router`] which returns a self-contained
//! `Router`.
//!
//! # Endpoints
//!
//! | Method | Path                      | Description                         |
//! |--------|---------------------------|-------------------------------------|
//! | GET    | `/world`                  | Full world snapshot                 |
//! | GET    | `/province/:id`           | Single province by ID               |
//! | GET    | `/faction/:id`            | Faction details + characters        |
//! | GET    | `/realm/:wallet`          | Player realm by wallet address      |
//! | GET    | `/turn/:number`           | Turn summary for a specific turn    |
//! | GET    | `/history/:province_id`   | Event history for a province        |
//! | GET    | `/stream`                 | SSE stream for real-time updates    |
//! | POST   | `/action`                 | Queue a game action                 |
//! | POST   | `/join`                   | Join the game as a faction          |

pub mod handlers;
pub mod events;
pub mod persistence;
pub mod streaming;

use std::sync::Arc;
use tokio::sync::RwLock;

use axum::routing::{get, post};
use axum::Router;

use crown_ash_sim::GameWorld;
use crown_ash_types::{QueuedAction, TurnSummary};
use serde::{Deserialize, Serialize};

// ─── Game State ────────────────────────────────────────────────────────────────

/// Complete game state managed by the API layer.
///
/// Wrapped in `Arc<RwLock<..>>` for concurrent read access from handlers
/// and exclusive write access from the simulation tick task.
///
/// The action queue lives here (not in `GameWorld`) because the simulation
/// engine drains queued actions during [`crown_ash_sim::tick`] and does not
/// persist them between ticks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrownAshGameState {
    /// The live game world (provinces, factions, characters, armies, etc.).
    pub world: GameWorld,
    /// Pending player actions to be processed on the next tick.
    pub action_queue: Vec<QueuedAction>,
    /// Accumulated turn summaries (newest last).
    pub turn_history: Vec<TurnSummary>,
}

impl CrownAshGameState {
    /// Create an empty game state with a default (uninitialized) world.
    pub fn empty() -> Self {
        Self {
            world: GameWorld {
                meta: crown_ash_types::WorldMeta::default(),
                provinces: Vec::new(),
                characters: Vec::new(),
                factions: Vec::new(),
                realms: Vec::new(),
                armies: Vec::new(),
                dynasties: Vec::new(),
                diplomacy: Vec::new(),
                action_queue: Vec::new(),
                plots: Vec::new(),
                trade_routes: Vec::new(),
                tombstones: Vec::new(),
                next_character_id: 0,
                next_army_id: 0,
                next_plot_id: 0,
                next_trade_route_id: 0,
                dirty: Default::default(),
            },
            action_queue: Vec::new(),
            turn_history: Vec::new(),
        }
    }
}

/// Shared handle to the game state, threaded through all route handlers.
pub type SharedGameState = Arc<RwLock<CrownAshGameState>>;

// ─── Router ────────────────────────────────────────────────────────────────────

/// Build the Crown & Ash router with SSE streaming support.
///
/// The caller provides a [`SharedGameState`] and an [`streaming::EventSender`]
/// for broadcasting real-time game events.
///
/// ```ignore
/// let game_state: SharedGameState = Arc::new(RwLock::new(CrownAshGameState::empty()));
/// let event_sender = streaming::create_event_channel(256);
/// let app = Router::new()
///     .nest("/api/v1/crown-ash", create_crown_ash_router(game_state, event_sender));
/// ```
pub fn create_crown_ash_router(
    game_state: SharedGameState,
    event_sender: streaming::EventSender,
) -> Router {
    let sse_state = streaming::SseState {
        sender: event_sender,
    };

    // REST routes use SharedGameState, SSE route uses SseState.
    let rest_routes = Router::new()
        .route("/world", get(handlers::get_world))
        .route("/province/{id}", get(handlers::get_province))
        .route("/faction/{id}", get(handlers::get_faction))
        .route("/realm/{wallet}", get(handlers::get_realm))
        .route("/turn/{number}", get(handlers::get_turn))
        .route("/history/{province_id}", get(handlers::get_province_history))
        .route("/action", post(handlers::submit_action))
        .route("/join", post(handlers::join_game))
        .with_state(game_state);

    let sse_route = Router::new()
        .route("/stream", get(streaming::sse_stream))
        .with_state(sse_state);

    rest_routes.merge(sse_route)
}
