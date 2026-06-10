//! Action submission system — player action buttons that POST to the server.
//!
//! Renders action buttons in an egui window (bottom-left) organized by category:
//! - **Province**: Raise Army, Build Improvement, Set Tax, Trade Routes, Convert
//! - **Army**: Move, Disband (when army is selected)
//! - **Diplomacy**: Declare War, Propose Treaty, Accept Treaty
//! - **Characters**: Assign Councilor, Designate Heir, Arrange Marriage
//! - **Intrigue**: Launch Plot, Back Plot, Investigate

use bevy::prelude::*;

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
    /// Tax slider value (0..100, mapped to FixedPoint 0..1000).
    pub tax_slider: f32,
}

impl Default for ActionState {
    fn default() -> Self {
        Self {
            last_result: None,
            submitting: false,
            pending: Arc::new(Mutex::new(None)),
            tax_slider: 20.0,
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
            drop(lock);
            action_state.submitting = false;
            action_state.last_result = Some(result);
        }
    }

    let ctx = contexts.ctx_mut();

    egui::Window::new("Actions")
        .anchor(egui::Align2::LEFT_BOTTOM, [10.0, -170.0])
        .default_width(240.0)
        .resizable(false)
        .collapsible(true)
        .show(ctx, |ui| {
            let Some(ref world) = game_state.world else {
                ui.label("No world data.");
                return;
            };

            let disabled = action_state.submitting;

            // ═══════════════════════════════════════════════════════
            // Province Actions (when a province is selected)
            // ═══════════════════════════════════════════════════════
            if let Some(pid) = selection.province {
                let prov_name = world.provinces.iter()
                    .find(|p| p.id == pid)
                    .map(|p| p.name.as_str())
                    .unwrap_or("?");
                ui.heading(format!("{} ({})", prov_name, pid));
                ui.separator();

                // Raise Army
                ui.add_enabled_ui(!disabled, |ui| {
                    if ui.button("Raise Army").clicked() {
                        fire_action(&config.server_url, serde_json::json!({
                            "action": { "RaiseArmy": { "province": pid } }
                        }), &mut action_state);
                    }
                });

                // Build Improvement
                ui.collapsing("Build Improvement", |ui| {
                    for imp in &[
                        "Market", "Temple", "Farmstead", "Mine", "Port",
                        "Lumbercamp", "Quarry", "Stables", "Walls",
                    ] {
                        ui.add_enabled_ui(!disabled, |ui| {
                            if ui.button(*imp).clicked() {
                                fire_action(&config.server_url, serde_json::json!({
                                    "action": { "BuildImprovement": {
                                        "province": pid,
                                        "improvement": imp
                                    }}
                                }), &mut action_state);
                            }
                        });
                    }
                });

                // Set Tax Rate
                ui.collapsing("Set Tax Rate", |ui| {
                    ui.add(egui::Slider::new(&mut action_state.tax_slider, 0.0..=100.0)
                        .suffix("%")
                        .text("Tax"));
                    ui.add_enabled_ui(!disabled, |ui| {
                        if ui.button("Apply Tax Rate").clicked() {
                            let rate = (action_state.tax_slider * 10.0) as i32; // 0..1000
                            fire_action(&config.server_url, serde_json::json!({
                                "action": { "SetTaxRate": {
                                    "province": pid,
                                    "rate": rate
                                }}
                            }), &mut action_state);
                        }
                    });
                });

                // Trade Routes
                if let Some(prov) = world.provinces.iter().find(|p| p.id == pid) {
                    ui.collapsing("Trade Route to...", |ui| {
                        for &neighbor in &prov.neighbors {
                            let label = world.provinces.iter()
                                .find(|p| p.id == neighbor)
                                .map(|p| p.name.as_str())
                                .unwrap_or("?");
                            ui.add_enabled_ui(!disabled, |ui| {
                                if ui.button(format!("{} ({})", label, neighbor)).clicked() {
                                    fire_action(&config.server_url, serde_json::json!({
                                        "action": { "EstablishTradeRoute": {
                                            "from": pid,
                                            "to": neighbor
                                        }}
                                    }), &mut action_state);
                                }
                            });
                        }
                    });
                }

                // Convert Province (religion)
                ui.collapsing("Convert Religion", |ui| {
                    for religion in &[
                        "EmberFaith", "FrostCult", "OldGods", "SaltMysticism",
                        "ShadowCrescent", "SteppeSpirits",
                    ] {
                        ui.add_enabled_ui(!disabled, |ui| {
                            if ui.button(*religion).clicked() {
                                fire_action(&config.server_url, serde_json::json!({
                                    "action": { "ConvertProvince": {
                                        "province": pid,
                                        "religion": religion
                                    }}
                                }), &mut action_state);
                            }
                        });
                    }
                });

                // ═══════════════════════════════════════════════════
                // Army Actions (armies in the selected province)
                // ═══════════════════════════════════════════════════
                let armies_here: Vec<_> = world.armies.iter()
                    .filter(|a| a.location == pid)
                    .collect();

                if !armies_here.is_empty() {
                    ui.separator();
                    ui.heading("Armies");

                    for army in &armies_here {
                        let owner_name = world.factions.iter()
                            .find(|f| f.id == army.owner_faction)
                            .map(|f| f.name.as_str())
                            .unwrap_or("?");
                        let total = army.troops.levy + army.troops.men_at_arms + army.troops.knights as u32;

                        ui.collapsing(format!("Army #{} [{}] ({})", army.id, owner_name, total), |ui| {
                            ui.label(format!("Levy: {} MaA: {} Knights: {}",
                                army.troops.levy, army.troops.men_at_arms, army.troops.knights));

                            // Move Army — show neighbor provinces
                            if let Some(prov) = world.provinces.iter().find(|p| p.id == pid) {
                                ui.label("Move to:");
                                for &neighbor in &prov.neighbors {
                                    let dest_name = world.provinces.iter()
                                        .find(|p| p.id == neighbor)
                                        .map(|p| p.name.as_str())
                                        .unwrap_or("?");
                                    ui.add_enabled_ui(!disabled, |ui| {
                                        if ui.button(format!("  {} ({})", dest_name, neighbor)).clicked() {
                                            fire_action(&config.server_url, serde_json::json!({
                                                "action": { "MoveArmy": {
                                                    "army": army.id,
                                                    "target": neighbor
                                                }}
                                            }), &mut action_state);
                                        }
                                    });
                                }
                            }

                            // Disband Army
                            ui.add_enabled_ui(!disabled, |ui| {
                                if ui.button("Disband").clicked() {
                                    fire_action(&config.server_url, serde_json::json!({
                                        "action": { "DisbandArmy": { "army": army.id } }
                                    }), &mut action_state);
                                }
                            });
                        });
                    }
                }
            }

            // ═══════════════════════════════════════════════════════
            // Diplomacy
            // ═══════════════════════════════════════════════════════
            ui.separator();
            ui.heading("Diplomacy");

            let alive_factions: Vec<_> = world.factions.iter()
                .filter(|f| f.alive)
                .collect();

            ui.collapsing("Declare War", |ui| {
                for casus in &["Conquest", "HolyWar", "Reconquest", "Insult"] {
                    ui.collapsing(format!("CB: {}", casus), |ui| {
                        for f in &alive_factions {
                            ui.add_enabled_ui(!disabled, |ui| {
                                if ui.button(format!("{} ({})", f.name, f.id)).clicked() {
                                    fire_action(&config.server_url, serde_json::json!({
                                        "action": { "DeclareWar": {
                                            "target": f.id,
                                            "casus_belli": casus
                                        }}
                                    }), &mut action_state);
                                }
                            });
                        }
                    });
                }
            });

            ui.collapsing("Propose Treaty", |ui| {
                for treaty in &[
                    "WhitePeace", "NonAggression", "DefensiveAlliance",
                    "TradeAgreement", "Marriage", "Surrender",
                ] {
                    ui.collapsing(format!("{}", treaty), |ui| {
                        for f in &alive_factions {
                            ui.add_enabled_ui(!disabled, |ui| {
                                if ui.button(format!("{} ({})", f.name, f.id)).clicked() {
                                    fire_action(&config.server_url, serde_json::json!({
                                        "action": { "ProposeTreaty": {
                                            "target": f.id,
                                            "treaty": treaty
                                        }}
                                    }), &mut action_state);
                                }
                            });
                        }
                    });
                }
            });

            // ═══════════════════════════════════════════════════════
            // Characters & Intrigue (when a faction is selected)
            // ═══════════════════════════════════════════════════════
            if let Some(fid) = selection.faction {
                let faction_chars: Vec<_> = world.characters.iter()
                    .filter(|c| c.faction == fid && c.alive)
                    .collect();

                if !faction_chars.is_empty() {
                    ui.separator();
                    ui.heading("Characters");

                    // Assign Councilor
                    ui.collapsing("Assign Councilor", |ui| {
                        for role in &["Marshal", "Chaplain", "Steward", "Spymaster"] {
                            ui.collapsing(format!("{}", role), |ui| {
                                for c in &faction_chars {
                                    ui.add_enabled_ui(!disabled, |ui| {
                                        if ui.button(format!("{} ({})", c.name, c.id)).clicked() {
                                            fire_action(&config.server_url, serde_json::json!({
                                                "action": { "AssignCouncilor": {
                                                    "character": c.id,
                                                    "role": role
                                                }}
                                            }), &mut action_state);
                                        }
                                    });
                                }
                            });
                        }
                    });

                    // Designate Heir
                    ui.collapsing("Designate Heir", |ui| {
                        for c in &faction_chars {
                            ui.add_enabled_ui(!disabled, |ui| {
                                if ui.button(format!("{} (age {})", c.name, c.age)).clicked() {
                                    fire_action(&config.server_url, serde_json::json!({
                                        "action": { "DesignateHeir": {
                                            "character": c.id
                                        }}
                                    }), &mut action_state);
                                }
                            });
                        }
                    });

                    // Arrange Marriage
                    ui.collapsing("Arrange Marriage", |ui| {
                        // Show all living characters across all factions as potential partners.
                        let all_chars: Vec<_> = world.characters.iter()
                            .filter(|c| c.alive)
                            .collect();
                        for own in &faction_chars {
                            ui.collapsing(format!("{}", own.name), |ui| {
                                for partner in &all_chars {
                                    if partner.id == own.id || partner.faction == fid {
                                        continue;
                                    }
                                    let partner_faction = world.factions.iter()
                                        .find(|f| f.id == partner.faction)
                                        .map(|f| f.name.as_str())
                                        .unwrap_or("?");
                                    ui.add_enabled_ui(!disabled, |ui| {
                                        if ui.button(format!("{} [{}]", partner.name, partner_faction)).clicked() {
                                            fire_action(&config.server_url, serde_json::json!({
                                                "action": { "ArrangeMarriage": {
                                                    "a": own.id,
                                                    "b": partner.id
                                                }}
                                            }), &mut action_state);
                                        }
                                    });
                                }
                            });
                        }
                    });

                    // Intrigue — Launch Plot
                    ui.separator();
                    ui.heading("Intrigue");

                    ui.collapsing("Launch Plot", |ui| {
                        for plot_type in &["Assassination", "Abduction", "Sabotage", "Scandal"] {
                            ui.collapsing(format!("{}", plot_type), |ui| {
                                // Targets: characters in other factions
                                let targets: Vec<_> = world.characters.iter()
                                    .filter(|c| c.alive && c.faction != fid)
                                    .collect();
                                for target in &targets {
                                    let tfaction = world.factions.iter()
                                        .find(|f| f.id == target.faction)
                                        .map(|f| f.name.as_str())
                                        .unwrap_or("?");
                                    ui.add_enabled_ui(!disabled, |ui| {
                                        if ui.button(format!("{} [{}]", target.name, tfaction)).clicked() {
                                            fire_action(&config.server_url, serde_json::json!({
                                                "action": { "LaunchPlot": {
                                                    "target": target.id,
                                                    "plot_type": plot_type
                                                }}
                                            }), &mut action_state);
                                        }
                                    });
                                }
                            });
                        }
                    });

                    // Investigate Plots (use spymaster)
                    ui.collapsing("Investigate Plots", |ui| {
                        for c in &faction_chars {
                            ui.add_enabled_ui(!disabled, |ui| {
                                if ui.button(format!("Send {} to investigate", c.name)).clicked() {
                                    fire_action(&config.server_url, serde_json::json!({
                                        "action": { "InvestigatePlot": {
                                            "spymaster": c.id
                                        }}
                                    }), &mut action_state);
                                }
                            });
                        }
                    });
                }
            }

            // ═══════════════════════════════════════════════════════
            // Status feedback
            // ═══════════════════════════════════════════════════════
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

    // Spawn a dedicated thread with its own tokio runtime.
    // Bevy's IoTaskPool is NOT a Tokio runtime, so reqwest panics there.
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build();
        let result = match rt {
            Ok(rt) => rt.block_on(post_action(&url, body)),
            Err(e) => ActionResult::Error(format!("Runtime error: {}", e)),
        };
        if let Ok(mut lock) = slot.lock() {
            *lock = Some(result);
        }
    });
}

async fn post_action(url: &str, body: serde_json::Value) -> ActionResult {
    let client = reqwest::Client::new();
    match client.post(url).json(&body).send().await {
        Ok(resp) => {
            if resp.status().is_success() {
                ActionResult::Success("Action submitted".to_string())
            } else {
                let status = resp.status();
                let body_text = resp.text().await.unwrap_or_default();
                ActionResult::Error(format!("Server {}: {}", status, body_text))
            }
        }
        Err(e) => ActionResult::Error(format!("Network error: {}", e)),
    }
}
