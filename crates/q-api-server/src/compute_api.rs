//! v9.5.0: Starship Endgame — Compute Orchestrator API
//!
//! Endpoints for monitoring and controlling the 8-layer compute scheduler.
//!
//! - GET  /api/v1/compute/status    → Full compute status (mode, resources, layers, trainer)
//! - POST /api/v1/compute/mode      → Change compute mode at runtime
//! - GET  /api/v1/compute/resources → Current resource snapshot (CPU, GPU, RAM, NET, DISK)
//! - GET  /api/v1/compute/trainer   → Trainer status with active cheats and boost %
//! - POST /api/v1/compute/trainer/toggle → Toggle individual trainer cheat

use axum::{
    extract::State,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn};

use crate::handlers::ApiResponse;
use crate::AppState;

/// Request body for changing compute mode
#[derive(Debug, Deserialize)]
pub struct SetModeRequest {
    pub mode: String,
}

/// Request body for toggling a trainer cheat
#[derive(Debug, Deserialize)]
pub struct ToggleCheatRequest {
    /// Cheat key: "F1" through "F10", or "all"
    pub cheat: String,
    /// true = activate, false = deactivate
    pub active: bool,
}

/// Response for trainer status
#[derive(Debug, Serialize)]
pub struct TrainerStatus {
    pub active: bool,
    pub cheats: Vec<CheatStatus>,
    pub estimated_boost_pct: f32,
    pub os_tuning: Vec<(String, String)>,
}

/// Individual cheat status
#[derive(Debug, Serialize)]
pub struct CheatStatus {
    pub key: String,
    pub name: String,
    pub active: bool,
    pub boost_pct: f32,
}

/// Create compute API router
pub fn create_compute_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/status", get(get_compute_status))
        .route("/mode", post(set_compute_mode))
        .route("/resources", get(get_resources))
        .route("/trainer", get(get_trainer_status))
        .route("/trainer/toggle", post(toggle_cheat))
}

/// GET /api/v1/compute/status — Full compute dashboard data
async fn get_compute_status(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse<serde_json::Value>> {
    match &state.compute_orchestrator {
        Some(orch) => {
            let status = orch.status();
            Json(ApiResponse::success(
                serde_json::to_value(status).unwrap_or_default(),
            ))
        }
        None => Json(ApiResponse::success(serde_json::json!({
            "enabled": false,
            "message": "Compute orchestrator not initialized"
        }))),
    }
}

/// POST /api/v1/compute/mode — Change compute mode at runtime
async fn set_compute_mode(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SetModeRequest>,
) -> Json<ApiResponse<serde_json::Value>> {
    let Some(orch) = &state.compute_orchestrator else {
        return Json(ApiResponse::error("Compute orchestrator not initialized".to_string()));
    };

    match req.mode.parse::<q_compute::ComputeMode>() {
        Ok(mode) => {
            let old_mode = orch.mode();
            orch.set_mode(mode);

            // If switching to NUKE, activate all trainer cheats
            if mode == q_compute::ComputeMode::Nuke {
                orch.trainer().activate_all();
            }

            info!(
                "🚀 [STARSHIP] Compute mode changed: {:?} → {:?}",
                old_mode, mode
            );

            Json(ApiResponse::success(serde_json::json!({
                "previous_mode": format!("{:?}", old_mode),
                "current_mode": format!("{:?}", mode),
                "trainer_active": mode == q_compute::ComputeMode::Nuke,
            })))
        }
        Err(e) => Json(ApiResponse::error(format!(
            "Invalid mode: {}. Use: mining-only, eco, full, nuke",
            e
        ))),
    }
}

/// GET /api/v1/compute/resources — Current resource snapshot
async fn get_resources(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse<serde_json::Value>> {
    let Some(orch) = &state.compute_orchestrator else {
        return Json(ApiResponse::error("Compute orchestrator not initialized".to_string()));
    };

    let snapshot = orch.monitor().snapshot();
    Json(ApiResponse::success(
        serde_json::to_value(snapshot).unwrap_or_default(),
    ))
}

/// GET /api/v1/compute/trainer — Trainer status with all cheats
async fn get_trainer_status(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse<serde_json::Value>> {
    let Some(orch) = &state.compute_orchestrator else {
        return Json(ApiResponse::error("Compute orchestrator not initialized".to_string()));
    };

    let trainer = orch.trainer();
    let cheats = vec![
        CheatStatus {
            key: "F1".into(),
            name: "INFINITE CORES".into(),
            active: trainer.infinite_cores.load(std::sync::atomic::Ordering::Relaxed),
            boost_pct: 150.0,
        },
        CheatStatus {
            key: "F2".into(),
            name: "GOD MODE MEMORY".into(),
            active: trainer.god_mode_memory.load(std::sync::atomic::Ordering::Relaxed),
            boost_pct: 30.0,
        },
        CheatStatus {
            key: "F3".into(),
            name: "SPEED HACK x100".into(),
            active: trainer.speed_hack.load(std::sync::atomic::Ordering::Relaxed),
            boost_pct: 400.0,
        },
        CheatStatus {
            key: "F4".into(),
            name: "WALL HACK".into(),
            active: trainer.wall_hack.load(std::sync::atomic::Ordering::Relaxed),
            boost_pct: 0.0,
        },
        CheatStatus {
            key: "F5".into(),
            name: "AIM BOT".into(),
            active: trainer.aim_bot.load(std::sync::atomic::Ordering::Relaxed),
            boost_pct: 0.0,
        },
        CheatStatus {
            key: "F6".into(),
            name: "NO CLIP".into(),
            active: trainer.no_clip.load(std::sync::atomic::Ordering::Relaxed),
            boost_pct: 50.0,
        },
        CheatStatus {
            key: "F7".into(),
            name: "INFINITE AMMO".into(),
            active: trainer.infinite_ammo.load(std::sync::atomic::Ordering::Relaxed),
            boost_pct: 0.0,
        },
        CheatStatus {
            key: "F8".into(),
            name: "RAPID FIRE".into(),
            active: trainer.rapid_fire.load(std::sync::atomic::Ordering::Relaxed),
            boost_pct: 20.0,
        },
        CheatStatus {
            key: "F9".into(),
            name: "TELEPORT".into(),
            active: trainer.teleport.load(std::sync::atomic::Ordering::Relaxed),
            boost_pct: 40.0,
        },
        CheatStatus {
            key: "F10".into(),
            name: "PRESTIGE MODE".into(),
            active: trainer.prestige_mode.load(std::sync::atomic::Ordering::Relaxed),
            boost_pct: 15.0,
        },
    ];

    let os_tuning = q_compute::os_tuner::OsTuner::status();

    let status = TrainerStatus {
        active: !trainer.active_cheats().is_empty(),
        cheats,
        estimated_boost_pct: trainer.estimated_boost_pct(),
        os_tuning,
    };

    Json(ApiResponse::success(
        serde_json::to_value(status).unwrap_or_default(),
    ))
}

/// POST /api/v1/compute/trainer/toggle — Toggle individual cheat
async fn toggle_cheat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ToggleCheatRequest>,
) -> Json<ApiResponse<serde_json::Value>> {
    use std::sync::atomic::Ordering;

    let Some(orch) = &state.compute_orchestrator else {
        return Json(ApiResponse::error("Compute orchestrator not initialized".to_string()));
    };

    let trainer = orch.trainer();

    match req.cheat.to_uppercase().as_str() {
        "F1" => trainer.infinite_cores.store(req.active, Ordering::SeqCst),
        "F2" => trainer.god_mode_memory.store(req.active, Ordering::SeqCst),
        "F3" => trainer.speed_hack.store(req.active, Ordering::SeqCst),
        "F4" => trainer.wall_hack.store(req.active, Ordering::SeqCst),
        "F5" => trainer.aim_bot.store(req.active, Ordering::SeqCst),
        "F6" => trainer.no_clip.store(req.active, Ordering::SeqCst),
        "F7" => trainer.infinite_ammo.store(req.active, Ordering::SeqCst),
        "F8" => trainer.rapid_fire.store(req.active, Ordering::SeqCst),
        "F9" => trainer.teleport.store(req.active, Ordering::SeqCst),
        "F10" => trainer.prestige_mode.store(req.active, Ordering::SeqCst),
        "ALL" => {
            if req.active {
                trainer.activate_all();
            } else {
                trainer.deactivate_all();
            }
        }
        _ => {
            return Json(ApiResponse::error(format!(
                "Unknown cheat: {}. Use F1-F10 or ALL",
                req.cheat
            )));
        }
    }

    let action = if req.active { "activated" } else { "deactivated" };
    info!("🎮 [TRAINER] {} {} via API", req.cheat, action);

    Json(ApiResponse::success(serde_json::json!({
        "cheat": req.cheat,
        "active": req.active,
        "active_cheats": trainer.active_cheats(),
        "estimated_boost_pct": trainer.estimated_boost_pct(),
    })))
}
