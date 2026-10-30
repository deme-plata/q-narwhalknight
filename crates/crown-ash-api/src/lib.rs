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
//! | POST   | `/action`                 | Queue a game action (signed)        |
//! | POST   | `/join`                   | Join the game as a faction (signed) |
//!
//! The two POST routes require an `X-Wallet-Auth` signature by the wallet they
//! name, bound to the body; see [`auth`].

pub mod auth;
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
        // axum 0.7 captures are `:name`; `{name}` is 0.8 syntax and matched only the
        // literal text, so these five routes answered 404 until v10.11.114h.
        .route("/province/:id", get(handlers::get_province))
        .route("/faction/:id", get(handlers::get_faction))
        .route("/realm/:wallet", get(handlers::get_realm))
        .route("/turn/:number", get(handlers::get_turn))
        .route("/history/:province_id", get(handlers::get_province_history))
        .route("/action", post(handlers::submit_action))
        .route("/join", post(handlers::join_game))
        .with_state(game_state);

    let sse_route = Router::new()
        .route("/stream", get(streaming::sse_stream))
        .with_state(sse_state);

    rest_routes.merge(sse_route)
}

#[cfg(test)]
mod router_tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use ed25519_dalek::{Signer, SigningKey};
    use http_body_util::BodyExt;
    use sha3::{Digest, Sha3_256};
    use tower::ServiceExt;

    const PREFIX: &str = "/api/v1/crown-ash";

    fn app(state: SharedGameState) -> Router {
        Router::new().nest(PREFIX, create_crown_ash_router(state, streaming::create_event_channel(16)))
    }

    fn wallet_of(sk: &SigningKey) -> String {
        auth::canonical_wallet(&sk.verifying_key().to_bytes())
    }

    /// The real 25-province map; faction 4 belongs to `owner`, faction 5 is free.
    fn world_with_owner(owner: &SigningKey) -> SharedGameState {
        let mut world = crown_ash_sim::world_gen::init_world(&crown_ash_types::WorldConfig::default(), [1; 32]);
        let w = wallet_of(owner);
        world.factions.iter_mut().find(|f| f.id == 4).unwrap().player_wallet = Some(w.clone());
        world.realms.iter_mut().find(|r| r.faction == 4).unwrap().owner_wallet = w;
        let mut gs = CrownAshGameState::empty();
        gs.world = world;
        Arc::new(RwLock::new(gs))
    }

    fn signed_post(path: &str, body: &str, signer: Option<&SigningKey>) -> Request<Body> {
        let full = format!("{PREFIX}{path}");
        let mut req = Request::post(&full).header("content-type", "application/json");
        if let Some(sk) = signer {
            let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
            let pk = sk.verifying_key().to_bytes();
            let body_hash = Sha3_256::digest(body.as_bytes());
            let mut h = Sha3_256::new();
            h.update(pk);
            h.update(ts.to_le_bytes());
            h.update(full.as_bytes());
            h.update(body_hash);
            let header = serde_json::json!({
                "address": wallet_of(sk), "timestamp": ts, "scheme": "Ed25519",
                "signature": hex::encode(sk.sign(&h.finalize()).to_bytes()),
                "body_hash": hex::encode(body_hash),
            });
            req = req.header(auth::AUTH_HEADER, header.to_string());
        }
        req.body(Body::from(body.to_string())).unwrap()
    }

    async fn send(app: &Router, req: Request<Body>) -> (StatusCode, String) {
        let res = app.clone().oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    async fn get(app: &Router, path: &str) -> StatusCode {
        let req = Request::get(format!("{PREFIX}{path}")).body(Body::empty()).unwrap();
        app.clone().oneshot(req).await.unwrap().status()
    }

    #[tokio::test]
    async fn parameter_routes_are_reachable() {
        let owner = SigningKey::from_bytes(&[4; 32]);
        let app = app(world_with_owner(&owner));
        // A bad number reaches the handler's Path parser (400); an unrouted path is 404.
        for p in ["/faction/abc", "/province/abc", "/turn/abc", "/history/abc"] {
            assert_eq!(get(&app, p).await, StatusCode::BAD_REQUEST, "{p}");
        }
        assert_eq!(get(&app, "/faction/4").await, StatusCode::OK);
        assert_eq!(get(&app, "/province/0").await, StatusCode::OK);
        assert_eq!(get(&app, &format!("/realm/{}", wallet_of(&owner))).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn unsigned_or_foreign_orders_are_refused_and_queue_nothing() {
        let owner = SigningKey::from_bytes(&[4; 32]);
        let thief = SigningKey::from_bytes(&[9; 32]);
        let state = world_with_owner(&owner);
        let app = app(state.clone());
        let body = format!(r#"{{"wallet":"{}","action":{{"RaiseArmy":{{"province":0}}}}}}"#, wallet_of(&owner));

        let (s, msg) = send(&app, signed_post("/action", &body, None)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "{msg}");
        let (s, msg) = send(&app, signed_post("/action", &body, Some(&thief))).await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{msg}");
        assert!(state.read().await.action_queue.is_empty());
    }

    #[tokio::test]
    async fn owner_order_is_queued_once() {
        let owner = SigningKey::from_bytes(&[4; 32]);
        let state = world_with_owner(&owner);
        let app = app(state.clone());
        // Upper-case hex in the body still names the owner; the queue stores the canonical form.
        let shouted = format!("qnk{}", wallet_of(&owner)[3..].to_uppercase());
        let body = format!(r#"{{"wallet":"{shouted}","action":{{"RaiseArmy":{{"province":0}}}},"nonce":"a"}}"#);
        let req = signed_post("/action", &body, Some(&owner));
        let replay = signed_post("/action", &body, Some(&owner));

        let (s, msg) = send(&app, req).await;
        assert_eq!(s, StatusCode::OK, "{msg}");
        let (s, msg) = send(&app, replay).await;
        assert_eq!(s, StatusCode::CONFLICT, "{msg}");

        let gs = state.read().await;
        assert_eq!(gs.action_queue.len(), 1);
        assert_eq!(gs.action_queue[0].wallet, wallet_of(&owner));
    }

    #[tokio::test]
    async fn join_needs_the_joining_wallet_signature() {
        let owner = SigningKey::from_bytes(&[4; 32]);
        let newcomer = SigningKey::from_bytes(&[5; 32]);
        let victim = SigningKey::from_bytes(&[6; 32]);
        let state = world_with_owner(&owner);
        let app = app(state.clone());

        // Claiming faction 5 in someone else's name.
        let body = format!(r#"{{"wallet":"{}","faction":5}}"#, wallet_of(&victim));
        let (s, msg) = send(&app, signed_post("/join", &body, Some(&newcomer))).await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{msg}");
        assert!(state.read().await.world.factions.iter().find(|f| f.id == 5).unwrap().player_wallet.is_none());

        let body = format!(r#"{{"wallet":"{}","faction":5}}"#, wallet_of(&newcomer));
        let (s, msg) = send(&app, signed_post("/join", &body, Some(&newcomer))).await;
        assert_eq!(s, StatusCode::OK, "{msg}");
        let gs = state.read().await;
        assert_eq!(
            gs.world.factions.iter().find(|f| f.id == 5).unwrap().player_wallet.as_deref(),
            Some(wallet_of(&newcomer).as_str())
        );
    }
}
