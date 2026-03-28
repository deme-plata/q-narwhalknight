//! Action submission system — player action buttons that POST to the server.
//!
//! Renders action buttons in an egui side panel (within the detail panel context)
//! and sends actions to the server via HTTP POST when clicked.

use bevy::prelude::*;
use bevy::tasks::IoTaskPool;
use bevy_egui::{egui, EguiContexts};
use std::sync::{Arc, Mutex};

use crate::resources::config::CrownAshConfig;
use crate::resources::game_state::ClientGameState;
use crate::resources::selection::Selection;

// ---------------------------------------------------------------------------
// Action submission state
// ---------------------------------------------------------------------------

/// Tracks pending action submissions and feedback messages.
#[derive(Resource)]
pub struct ActionState {
    /// Result of the last action submission (shown as toast).
    pub last_result: Option<ActionResult>,
    /// True while an action POST is in-flight.
    pub submitting: bool,
    /// Shared slot for async result delivery.
    pending: Arc<Mutex<Option<ActionResult>>>,
}

impl Default for ActionState {
    fn default() -> Self {
        Self {
            last_result: None,
            submitting: false,
            pending: Arc::new(Mutex::new(None)),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ActionResult {
    Success(String),
    Error(String),
}

// ---------------------------------------------------------------------------
// System: action buttons
// ---------------------------------------------------------------------------

pub fn action_buttons(
    mut contexts: EguiContexts,
    game_state: Res<ClientGameState>,
    selection: Res<Selection>,
    config: Res<CrownAshConfig>,
    mut action_state: ResMut<ActionState>,
) {
    // Drain any completed async result.
    let pending_clone = Arc::clone(&action_state.pending);
    if let Ok(mut lock) = pending_clone.try_lock() {
        if let Some(result) = lock.take() {
            drop(lock); // release mutex before mutating action_state
            action_state.submitting = false;
            action_state.last_result = Some(result);
        }
    }

    let ctx = contexts.ctx_mut();

    egui::Window::new("Actions")
        .anchor(egui::Align2::LEFT_BOTTOM, [10.0, -170.0])
        .default_width(220.0)
        .resizable(false)
        .collapsible(true)
        .show(ctx, |ui| {
            let Some(ref world) = game_state.world else {
                ui.label("No world data.");
                return;
            };

            let disabled = action_state.submitting;

            // --- Province actions (when a province is selected) ---
            if let Some(pid) = selection.province {
                ui.heading(format!("Province #{}", pid));
                ui.separator();

                // Raise Army
                ui.add_enabled_ui(!disabled, |ui| {
                    if ui.button("Raise Army").clicked() {
                        let body = serde_json::json!({
                            "action": { "RaiseArmy": { "province": pid } }
                        });
                        fire_action(&config.server_url, body, &mut action_state);
                    }
                });

                // Build Improvement (show a few common ones)
                ui.collapsing("Build Improvement", |ui| {
                    for imp in &["Market", "Temple", "Farmstead", "Mine", "Port", "Lumbercamp", "Quarry", "Stables", "Walls"] {
                        ui.add_enabled_ui(!disabled, |ui| {
                            if ui.button(*imp).clicked() {
                                let body = serde_json::json!({
                                    "action": { "BuildImprovement": {
                                        "province": pid,
                                        "improvement": imp
                                    }}
                                });
                                fire_action(&config.server_url, body, &mut action_state);
                            }
                        });
                    }
                });

                // Establish Trade Route (to neighbors)
                if let Some(prov) = world.provinces.iter().find(|p| p.id == pid) {
                    ui.collapsing("Trade Route to...", |ui| {
                        for &neighbor in &prov.neighbors {
                            let label = world.provinces.iter()
                                .find(|p| p.id == neighbor)
                                .map(|p| p.name.as_str())
                                .unwrap_or("?");
                            ui.add_enabled_ui(!disabled, |ui| {
                                if ui.button(format!("{} ({})", label, neighbor)).clicked() {
                                    let body = serde_json::json!({
                                        "action": { "EstablishTradeRoute": {
                                            "from": pid,
                                            "to": neighbor
                                        }}
                                    });
                                    fire_action(&config.server_url, body, &mut action_state);
                                }
                            });
                        }
                    });
                }
            }

            // --- Diplomacy actions ---
            ui.separator();
            ui.heading("Diplomacy");

            let alive_factions: Vec<_> = world.factions.iter()
                .filter(|f| f.alive)
                .collect();

            ui.collapsing("Declare War", |ui| {
                for f in &alive_factions {
                    ui.add_enabled_ui(!disabled, |ui| {
                        if ui.button(format!("{} ({})", f.name, f.id)).clicked() {
                            let body = serde_json::json!({
                                "action": { "DeclareWar": {
                                    "target": f.id,
                                    "casus_belli": "Conquest"
                                }}
                            });
                            fire_action(&config.server_url, body, &mut action_state);
                        }
                    });
                }
            });

            ui.collapsing("Propose Peace", |ui| {
                for f in &alive_factions {
                    ui.add_enabled_ui(!disabled, |ui| {
                        if ui.button(format!("{} ({})", f.name, f.id)).clicked() {
                            let body = serde_json::json!({
                                "action": { "ProposeTreaty": {
                                    "target": f.id,
                                    "treaty": "WhitePeace"
                                }}
                            });
                            fire_action(&config.server_url, body, &mut action_state);
                        }
                    });
                }
            });

            // --- Status feedback ---
            ui.separator();
            if action_state.submitting {
                ui.spinner();
                ui.label("Submitting...");
            }
            if let Some(ref result) = action_state.last_result {
                match result {
                    ActionResult::Success(msg) => {
                        ui.colored_label(egui::Color32::GREEN, msg);
                    }
                    ActionResult::Error(msg) => {
                        ui.colored_label(egui::Color32::RED, msg);
                    }
                }
            }
        });
}

// ---------------------------------------------------------------------------
// Async HTTP POST
// ---------------------------------------------------------------------------

fn fire_action(
    server_url: &str,
    body: serde_json::Value,
    state: &mut ResMut<ActionState>,
) {
    let url = format!("{}/crown-ash/action", server_url);
    let slot = Arc::clone(&state.pending);
    state.submitting = true;
    state.last_result = None;

    IoTaskPool::get()
        .spawn(async move {
            let result = post_action(&url, body).await;
            if let Ok(mut lock) = slot.lock() {
                *lock = Some(result);
            }
        })
        .detach();
}

async fn post_action(url: &str, body: serde_json::Value) -> ActionResult {
    let client = reqwest::Client::new();
    match client.post(url).json(&body).send().await {
        Ok(resp) => {
            if resp.status().is_success() {
                ActionResult::Success("Action submitted".to_string())
            } else {
                ActionResult::Error(format!("Server error: {}", resp.status()))
            }
        }
        Err(e) => ActionResult::Error(format!("Network error: {}", e)),
    }
}
