//! Network plugin — polls the Crown & Ash REST API and pushes snapshots into
//! [`ClientGameState`].
//!
//! Uses Bevy's [`IoTaskPool`] to fire non-blocking HTTP requests so the main
//! thread (and thus the renderer) is never stalled by network latency.

use bevy::prelude::*;
use bevy::tasks::IoTaskPool;
use std::sync::{Arc, Mutex};

use crate::resources::{
    config::CrownAshConfig,
    game_state::{ApiResponse, ClientGameState, ConnectionStatus, WorldSnapshot},
    selection::Selection,
};

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

/// Bevy plugin that handles all server communication for Crown & Ash.
///
/// # Systems
/// - `poll_server` (Update) — fires a new HTTP request every
///   `CrownAshConfig::poll_interval_secs` seconds and applies the response to
///   `ClientGameState`.
pub struct CrownAshNetworkPlugin;

impl Plugin for CrownAshNetworkPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ClientGameState>()
            .init_resource::<CrownAshConfig>()
            .init_resource::<Selection>()
            .init_resource::<NetworkState>()
            .add_systems(Startup, setup_network)
            .add_systems(Update, poll_server);
    }
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

/// Internal bookkeeping for the polling loop.
///
/// `pending` is shared with the async task: the task writes `Some(result)` and
/// the `poll_server` system drains it on the next frame.
#[derive(Resource)]
struct NetworkState {
    timer: Timer,
    pending: Arc<Mutex<Option<Result<WorldSnapshot, String>>>>,
    /// True while an HTTP request is in-flight (prevents overlapping fetches).
    in_flight: bool,
}

impl Default for NetworkState {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(5.0, TimerMode::Repeating),
            pending: Arc::new(Mutex::new(None)),
            in_flight: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// One-shot startup system: synchronises the timer period with config and fires
/// the first request immediately.
fn setup_network(config: Res<CrownAshConfig>, mut net: ResMut<NetworkState>) {
    net.timer = Timer::from_seconds(config.poll_interval_secs, TimerMode::Repeating);
    // Fire the first fetch immediately on startup.
    fire_request(&config.server_url, &net.pending);
    net.in_flight = true;
}

/// Per-frame system: check for completed requests, then fire new ones on timer.
fn poll_server(
    time: Res<Time>,
    config: Res<CrownAshConfig>,
    mut net: ResMut<NetworkState>,
    mut game_state: ResMut<ClientGameState>,
) {
    // ------------------------------------------------------------------
    // 1. Drain the completed result (if any).
    // ------------------------------------------------------------------
    if let Ok(mut lock) = net.pending.try_lock() {
        if let Some(result) = lock.take() {
            net.in_flight = false;
            match result {
                Ok(snapshot) => {
                    game_state.last_update_turn = snapshot.meta.turn;
                    game_state.connection = ConnectionStatus::Connected;
                    game_state.world = Some(snapshot);
                }
                Err(e) => {
                    game_state.connection = ConnectionStatus::Error(e);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 2. Fire a new request when the timer ticks (skip if one is already
    //    in-flight to avoid request pileup on slow connections).
    // ------------------------------------------------------------------
    net.timer.tick(time.delta());
    if net.timer.just_finished() && !net.in_flight {
        fire_request(&config.server_url, &net.pending);
        net.in_flight = true;

        if game_state.connection == ConnectionStatus::Disconnected {
            game_state.connection = ConnectionStatus::Connecting;
        }
    }
}

// ---------------------------------------------------------------------------
// Async helpers
// ---------------------------------------------------------------------------

/// Spawn an async HTTP GET on the IO task pool.
fn fire_request(
    server_url: &str,
    pending: &Arc<Mutex<Option<Result<WorldSnapshot, String>>>>,
) {
    let url = format!("{}/crown-ash/world", server_url);
    let slot = Arc::clone(pending);

    IoTaskPool::get()
        .spawn(async move {
            let result = fetch_world(&url).await;
            if let Ok(mut lock) = slot.lock() {
                *lock = Some(result);
            }
        })
        .detach();
}

/// Perform the actual HTTP fetch and deserialize the JSON envelope.
async fn fetch_world(url: &str) -> Result<WorldSnapshot, String> {
    let resp = reqwest::get(url).await.map_err(|e| format!("HTTP error: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("Server returned {}", resp.status()));
    }

    let api: ApiResponse<WorldSnapshot> = resp
        .json()
        .await
        .map_err(|e| format!("JSON decode error: {e}"))?;

    if api.success {
        api.data.ok_or_else(|| "Response success=true but data was null".to_string())
    } else {
        Err(api.error.unwrap_or_else(|| "Unknown server error".to_string()))
    }
}
