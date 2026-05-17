//! egui-based UI panels for the Crown & Ash client.
//!
//! Systems:
//! - `top_bar` — turn counter, faction count, population, connection indicator
//! - `faction_list` — left panel with all factions overview
//! - `detail_panel` — province / faction / character / army details
//! - `event_feed` — scrolling narrative event log
//! - `minimap` — small map overview with clickable provinces
//! - `join_dialog` — "Join Game" overlay when the player has no realm
//! - `dialog_bubbles` — floating speech bubbles for Tier 2/3 LLM-generated text
//! - `keyboard_shortcuts` — ESC to deselect, Tab to cycle factions

use bevy::prelude::*;
use bevy::input::ButtonInput;
use bevy_egui::{egui, EguiContexts};
use std::sync::{Arc, Mutex};

use crate::resources::config::CrownAshConfig;
use crate::resources::game_state::{ClientGameState, ConnectionStatus};
use crate::resources::narrative_state::{DialogState, NarrativeImportance, NarrativeState, ToastState};
use crate::resources::selection::Selection;
use crown_ash_types::{FixedPoint, GameEvent};

// ---------------------------------------------------------------------------
// Join game state
// ---------------------------------------------------------------------------

/// OAuth2 device-login state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceLoginPhase {
    /// Not started — show "Login with Quillon Wallet" button.
    Idle,
    /// Requesting device code from server.
    Requesting,
    /// Waiting for user to approve in browser.
    WaitingForApproval {
        device_code: String,
        verification_url: String,
    },
    /// Login complete — wallet address received.
    Complete { wallet_address: String },
    /// Error during device login.
    Error(String),
}

/// Tracks the "Join Game" dialog and player identity.
#[derive(Resource)]
pub struct JoinState {
    /// Wallet address (populated by device-login or manual input).
    pub wallet_input: String,
    /// Selected faction in the join dialog (0-6).
    pub selected_faction: u8,
    /// True if the player has joined (or is the server operator).
    pub joined: bool,
    /// Join request in flight.
    pub joining: bool,
    /// Join result message.
    pub join_result: Option<String>,
    /// Async result slot for join request.
    pending: Arc<Mutex<Option<Result<String, String>>>>,
    /// OAuth2 device-login phase.
    pub device_login: DeviceLoginPhase,
    /// Async result slot for device-login polling.
    device_login_pending: Arc<Mutex<Option<Result<DeviceLoginPhase, String>>>>,
    /// Manual wallet input mode (fallback).
    pub manual_mode: bool,
}

impl Default for JoinState {
    fn default() -> Self {
        Self {
            wallet_input: String::new(),
            selected_faction: 0,
            joined: false,
            joining: false,
            join_result: None,
            pending: Arc::new(Mutex::new(None)),
            device_login: DeviceLoginPhase::Idle,
            device_login_pending: Arc::new(Mutex::new(None)),
            manual_mode: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: display a FixedPoint as "integer.fractional" (3 decimals)
// ---------------------------------------------------------------------------

fn fp_display(fp: FixedPoint) -> String {
    let raw = fp.raw();
    let sign = if raw < 0 { "-" } else { "" };
    let abs = raw.unsigned_abs() as i64;
    format!("{}{}.{:03}", sign, abs / 1000, abs % 1000)
}

// ---------------------------------------------------------------------------
// Top bar — always visible, spans the full window width.
// ---------------------------------------------------------------------------

pub fn top_bar(
    mut contexts: EguiContexts,
    game_state: Res<ClientGameState>,
    config: Res<CrownAshConfig>,
) {
    let ctx = contexts.ctx_mut();

    egui::TopBottomPanel::top("crown_ash_top_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            match &game_state.world {
                Some(world) => {
                    ui.label(format!("Turn: {}", world.meta.turn));
                    ui.separator();

                    let alive = world.factions.iter().filter(|f| f.alive).count();
                    ui.label(format!("{} factions alive", alive));
                    ui.separator();

                    let pop: u64 = world.provinces.iter().map(|p| p.population as u64).sum();
                    ui.label(format!("Pop: {}", format_population(pop)));
                    ui.separator();

                    ui.label(format!("Armies: {}", world.armies.len()));
                    ui.separator();

                    ui.label(format!("Actions queued: {}", world.action_queue_size));
                }
                None => {
                    ui.label("Waiting for server...");
                }
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let (icon, color) = match &game_state.connection {
                    ConnectionStatus::Connected => ("Connected", egui::Color32::GREEN),
                    ConnectionStatus::Connecting => ("Connecting...", egui::Color32::YELLOW),
                    ConnectionStatus::Disconnected => ("Disconnected", egui::Color32::RED),
                    ConnectionStatus::Error(_) => ("Error", egui::Color32::RED),
                };
                ui.colored_label(color, icon);
                ui.label(format!("Server: {}", config.server_url));
            });
        });
    });
}

// ---------------------------------------------------------------------------
// Detail panel — right side, shows selected entity info.
// ---------------------------------------------------------------------------

pub fn detail_panel(
    mut contexts: EguiContexts,
    game_state: Res<ClientGameState>,
    selection: Res<Selection>,
    narrative: Res<NarrativeState>,
) {
    let ctx = contexts.ctx_mut();

    egui::SidePanel::right("crown_ash_detail")
        .default_width(350.0)
        .min_width(280.0)
        .show(ctx, |ui| {
            let Some(ref world) = game_state.world else {
                ui.heading("No data");
                ui.label("Waiting for world snapshot...");
                return;
            };

            // Province detail
            if let Some(pid) = selection.province {
                if let Some(prov) = world.provinces.iter().find(|p| p.id == pid) {
                    ui.heading(&prov.name);
                    ui.label(format!("Province #{} ({:?})", prov.id, prov.terrain));
                    ui.separator();

                    let faction_name = world.factions.iter()
                        .find(|f| f.id == prov.controller)
                        .map(|f| f.name.as_str())
                        .unwrap_or("Unknown");
                    ui.label(format!("Controller: {} ({})", faction_name, prov.controller));
                    ui.label(format!("Population: {}", prov.population));
                    ui.label(format!("Prosperity: {}", fp_display(prov.prosperity)));
                    ui.label(format!("Unrest: {}", fp_display(prov.unrest)));
                    ui.label(format!("Fortification: {}", prov.fortification));
                    ui.label(format!("Tax rate: {}", fp_display(prov.tax_rate)));

                    // Garrison
                    let g = &prov.garrison;
                    ui.separator();
                    ui.label("Garrison:");
                    ui.label(format!(
                        "  Levy: {}  MaA: {}  Knights: {}",
                        g.levy, g.men_at_arms, g.knights
                    ));

                    // Resources
                    ui.separator();
                    ui.label("Resources:");
                    ui.label(format!("  Food: {}  Gold: {}", fp_display(prov.resources.food), fp_display(prov.resources.gold)));
                    ui.label(format!("  Iron: {}  Timber: {}", fp_display(prov.resources.iron), fp_display(prov.resources.timber)));
                    ui.label(format!("  Stone: {}  Horses: {}", fp_display(prov.resources.stone), fp_display(prov.resources.horses)));

                    // Improvements
                    if !prov.improvements.is_empty() {
                        ui.separator();
                        ui.label("Improvements:");
                        for imp in &prov.improvements {
                            ui.label(format!("  {:?}", imp));
                        }
                    }

                    // Construction queue
                    if !prov.construction_queue.is_empty() {
                        ui.separator();
                        ui.label("Under construction:");
                        for (imp, turns) in &prov.construction_queue {
                            ui.label(format!("  {:?} ({} turns left)", imp, turns));
                        }
                    }

                    // Armies in this province
                    let armies_here: Vec<_> = world.armies.iter()
                        .filter(|a| a.location == pid)
                        .collect();
                    if !armies_here.is_empty() {
                        ui.separator();
                        ui.label(format!("Armies ({}):", armies_here.len()));
                        for army in armies_here {
                            let owner = world.factions.iter()
                                .find(|f| f.id == army.owner_faction)
                                .map(|f| f.name.as_str())
                                .unwrap_or("?");
                            ui.label(format!(
                                "  #{} [{}] L:{} M:{} K:{}",
                                army.id, owner,
                                army.troops.levy, army.troops.men_at_arms, army.troops.knights
                            ));
                        }
                    }

                    // Province religion narrative
                    if let Some(rel_text) = narrative.province_religion_text(pid) {
                        if !rel_text.is_empty() {
                            ui.separator();
                            ui.label(egui::RichText::new("Faith").strong());
                            ui.label(egui::RichText::new(rel_text).italics().color(
                                egui::Color32::from_rgb(180, 160, 200),
                            ));
                        }
                    }

                    // Province siege history
                    {
                        let ctx = build_narrative_ctx(&game_state);
                        let siege_text = crown_ash_narrative::history::siege_narrative(
                            pid, &prov.name, &game_state.events, &ctx,
                        );
                        if !siege_text.is_empty() {
                            ui.separator();
                            ui.label(egui::RichText::new("Sieges").strong());
                            ui.label(egui::RichText::new(&siege_text).italics().color(
                                egui::Color32::from_rgb(200, 140, 140),
                            ));
                        }

                        // Province trade routes
                        let trade_text = crown_ash_narrative::history::trade_narrative(
                            pid, &prov.name, &game_state.events, &ctx,
                        );
                        if !trade_text.is_empty() {
                            ui.separator();
                            ui.label(egui::RichText::new("Trade").strong());
                            ui.label(egui::RichText::new(&trade_text).italics().color(
                                egui::Color32::from_rgb(160, 190, 140),
                            ));
                        }

                        // Province construction/improvements
                        let imp_strs: Vec<String> = prov.improvements.iter()
                            .map(|i| format!("{:?}", i))
                            .collect();
                        let imp_refs: Vec<&str> = imp_strs.iter()
                            .map(|s| s.as_str())
                            .collect();
                        let constr_text = crown_ash_narrative::history::construction_narrative(
                            pid, &prov.name, &imp_refs, &game_state.events, &ctx,
                        );
                        if !constr_text.is_empty() {
                            ui.separator();
                            ui.label(egui::RichText::new("Improvements").strong());
                            ui.label(egui::RichText::new(&constr_text).italics().color(
                                egui::Color32::from_rgb(170, 190, 170),
                            ));
                        }
                    }

                    // Province history (narrative)
                    if let Some(history) = narrative.province_history(pid) {
                        if !history.is_empty() {
                            ui.separator();
                            ui.label(egui::RichText::new("History").strong());
                            ui.label(history);
                        }
                    }
                }
            }

            // Faction detail (if selected)
            if let Some(fid) = selection.faction {
                ui.separator();
                ui.separator();
                if let Some(faction) = world.factions.iter().find(|f| f.id == fid) {
                    ui.heading(&faction.name);
                    ui.label(format!("Faction #{}", faction.id));
                    ui.label(if faction.alive { "Status: Alive" } else { "Status: Eliminated" });
                    ui.label(format!("Culture: {:?}", faction.culture));
                    ui.label(format!("Religion: {:?}", faction.religion));

                    let prov_count = world.provinces.iter()
                        .filter(|p| p.controller == fid)
                        .count();
                    ui.label(format!("Provinces: {}", prov_count));

                    let army_count = world.armies.iter()
                        .filter(|a| a.owner_faction == fid)
                        .count();
                    ui.label(format!("Armies: {}", army_count));

                    // Realm prosperity narrative
                    if let Some(prosperity) = narrative.realm_prosperity_text(fid) {
                        if !prosperity.is_empty() {
                            ui.separator();
                            ui.label(egui::RichText::new("State of the Realm").strong());
                            ui.colored_label(
                                egui::Color32::from_rgb(160, 190, 140),
                                egui::RichText::new(prosperity).italics(),
                            );
                        }
                    }

                    // Realm info
                    if let Some(realm) = world.realms.iter().find(|r| r.faction == fid) {
                        ui.separator();
                        ui.label(format!("Treasury: {}", fp_display(realm.treasury)));
                        ui.label(format!("Age: {} turns", realm.age));
                        if !realm.at_war_with.is_empty() {
                            let enemies: Vec<String> = realm.at_war_with.iter()
                                .map(|&eid| world.factions.iter()
                                    .find(|f| f.id == eid)
                                    .map(|f| f.name.clone())
                                    .unwrap_or_else(|| format!("#{}", eid)))
                                .collect();
                            ui.colored_label(
                                egui::Color32::from_rgb(220, 80, 80),
                                format!("At war with: {}", enemies.join(", ")),
                            );

                            // War summary prose for each active war
                            for &enemy_id in &realm.at_war_with {
                                if let Some(war_text) = narrative.war_summary(fid, enemy_id) {
                                    if !war_text.is_empty() {
                                        ui.add_space(4.0);
                                        ui.colored_label(
                                            egui::Color32::from_rgb(200, 160, 120),
                                            egui::RichText::new(war_text).italics(),
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Diplomatic relations with other factions
                    {
                        let other_factions: Vec<u8> = world.factions.iter()
                            .filter(|f| f.id != fid && f.alive)
                            .map(|f| f.id)
                            .collect();
                        if !other_factions.is_empty() {
                            ui.separator();
                            ui.label(egui::RichText::new("Diplomatic Relations").strong());
                            for &other_id in &other_factions {
                                if let Some(diplo_text) = narrative.diplomacy_text(fid, other_id) {
                                    if !diplo_text.is_empty() {
                                        ui.add_space(2.0);
                                        ui.label(egui::RichText::new(diplo_text).italics().color(
                                            egui::Color32::from_rgb(170, 180, 200),
                                        ));
                                    }
                                }
                            }
                        }
                    }

                    // Succession history
                    {
                        let ctx = build_narrative_ctx(&game_state);
                        let succ_text = crown_ash_narrative::history::succession_narrative(
                            fid, &game_state.events, &ctx,
                        );
                        if !succ_text.is_empty() {
                            ui.separator();
                            ui.label(egui::RichText::new("Succession").strong());
                            ui.label(egui::RichText::new(&succ_text).italics().color(
                                egui::Color32::from_rgb(200, 170, 130),
                            ));
                        }
                    }

                    // Characters in this faction
                    let chars: Vec<_> = world.characters.iter()
                        .filter(|c| c.faction == fid && c.alive)
                        .collect();
                    if !chars.is_empty() {
                        ui.separator();
                        ui.label(format!("Characters ({}):", chars.len()));
                        for c in chars.iter().take(10) {
                            ui.label(format!(
                                "  {} ({:?}, age {})",
                                c.name, c.role, c.age
                            ));
                        }
                        if chars.len() > 10 {
                            ui.label(format!("  ... and {} more", chars.len() - 10));
                        }
                    }

                    // Faction history (narrative)
                    if let Some(history) = narrative.faction_history(fid) {
                        if !history.is_empty() {
                            ui.separator();
                            ui.label(egui::RichText::new("History").strong());
                            ui.label(history);
                        }
                    }
                }
            }

            // Character detail
            if let Some(cid) = selection.character {
                ui.separator();
                ui.separator();
                if let Some(c) = world.characters.iter().find(|ch| ch.id == cid) {
                    ui.heading(&c.name);
                    ui.label(format!("Character #{} (age {})", c.id, c.age));
                    ui.label(format!("Role: {:?}", c.role));
                    ui.label(format!("Faction: {}", c.faction));
                    ui.label(if c.alive { "Alive" } else { "Dead" });
                    ui.separator();
                    ui.label("Stats:");
                    ui.label(format!("  Martial: {}", fp_display(c.stats.martial)));
                    ui.label(format!("  Diplomacy: {}", fp_display(c.stats.diplomacy)));
                    ui.label(format!("  Stewardship: {}", fp_display(c.stats.stewardship)));
                    ui.label(format!("  Intrigue: {}", fp_display(c.stats.intrigue)));
                    ui.label(format!("  Learning: {}", fp_display(c.stats.learning)));
                    ui.separator();
                    ui.label(format!("Health: {}", fp_display(c.health)));
                    ui.label(format!("Prestige: {}", fp_display(c.prestige)));
                    ui.label(format!("Legitimacy: {}", fp_display(c.legitimacy)));
                    if !c.traits.is_empty() {
                        ui.label(format!("Traits: {:?}", c.traits));
                    }

                    // Personality archetype
                    let archetype = crown_ash_narrative::personality::derive_archetype(&c.traits);
                    ui.label(format!("Personality: {}", archetype.label()));

                    // Relationships
                    if !c.relations.is_empty() {
                        let rel_tuples: Vec<(u32, Option<crown_ash_types::character::RelationType>, i64)> =
                            c.relations.iter()
                                .map(|r| (r.target, r.relation_type, r.opinion.0 as i64))
                                .collect();
                        let ctx = build_narrative_ctx(&game_state);
                        let rel_text = crown_ash_narrative::history::relationship_narrative(
                            cid, &rel_tuples, &game_state.events, &ctx,
                        );
                        if !rel_text.is_empty() {
                            ui.separator();
                            ui.label(egui::RichText::new("Relationships").strong());
                            ui.colored_label(
                                egui::Color32::from_rgb(200, 180, 160),
                                egui::RichText::new(&rel_text).italics(),
                            );
                        }
                    }

                    // Dynasty lineage
                    if c.dynasty > 0 {
                        let ctx = build_narrative_ctx(&game_state);
                        let lineage = crown_ash_narrative::history::dynasty_lineage(
                            cid, c.dynasty, &game_state.events, &ctx,
                        );
                        if !lineage.is_empty() {
                            ui.separator();
                            ui.label(egui::RichText::new("Lineage").strong());
                            ui.colored_label(
                                egui::Color32::from_rgb(180, 170, 200),
                                &lineage,
                            );
                        }
                    }

                    // Character biography
                    {
                        let ctx = build_narrative_ctx(&game_state);
                        let trait_strs: Vec<String> = c.traits.iter()
                            .map(|t| format!("{:?}", t))
                            .collect();
                        let trait_refs: Vec<&str> = trait_strs.iter()
                            .map(|s| s.as_str())
                            .collect();
                        let bio = crown_ash_narrative::history::character_biography(
                            cid, &c.name, c.age, &format!("{:?}", c.role),
                            c.faction, c.alive, &trait_refs,
                            &game_state.events, &ctx,
                        );
                        if !bio.is_empty() {
                            ui.separator();
                            ui.label(egui::RichText::new("Biography").strong());
                            ui.label(egui::RichText::new(&bio).italics().color(
                                egui::Color32::from_rgb(190, 185, 170),
                            ));
                        }
                    }

                    // Character Chronicle (life history)
                    if let Some(chronicle_text) = narrative.chronicle_text(cid) {
                        if !chronicle_text.is_empty() {
                            ui.separator();
                            ui.label(egui::RichText::new("Chronicle").strong());
                            egui::ScrollArea::vertical()
                                .id_salt("character_chronicle")
                                .max_height(200.0)
                                .show(ui, |ui| {
                                    ui.label(&chronicle_text);
                                });
                        }
                    }
                }
            }

            // Army detail
            if let Some(aid) = selection.army {
                ui.separator();
                ui.separator();
                if let Some(army) = world.armies.iter().find(|a| a.id == aid) {
                    ui.heading(format!("Army #{}", army.id));
                    let owner = world.factions.iter()
                        .find(|f| f.id == army.owner_faction)
                        .map(|f| f.name.as_str())
                        .unwrap_or("Unknown");
                    ui.label(format!("Owner: {}", owner));
                    ui.label(format!("Location: province {}", army.location));
                    if let Some(dest) = army.destination {
                        ui.label(format!("Moving to: province {}", dest));
                    }
                    if !army.movement_queue.is_empty() {
                        let path: Vec<String> = army.movement_queue.iter()
                            .map(|p| format!("{}", p))
                            .collect();
                        ui.label(format!("Path: {}", path.join(" -> ")));
                    }
                    if army.siege.is_some() {
                        ui.colored_label(
                            egui::Color32::from_rgb(220, 160, 50),
                            "Besieging...",
                        );
                    }
                    ui.separator();
                    ui.label(format!("Levy: {}", army.troops.levy));
                    ui.label(format!("Men at Arms: {}", army.troops.men_at_arms));
                    ui.label(format!("Knights: {}", army.troops.knights));
                    ui.separator();
                    ui.label(format!("Morale: {}", fp_display(army.morale)));
                    ui.label(format!("Supply: {}", fp_display(army.supply)));
                    ui.label(format!("Raised turn: {}", army.raised_turn));

                    // Army narrative
                    {
                        let ctx = build_narrative_ctx(&game_state);
                        let army_text = crown_ash_narrative::history::army_narrative(
                            army.id, army.owner_faction, army.commander,
                            army.location,
                            army.troops.levy, army.troops.men_at_arms, army.troops.knights,
                            army.morale.0, army.raised_turn,
                            &game_state.events, &ctx,
                        );
                        if !army_text.is_empty() {
                            ui.separator();
                            ui.label(egui::RichText::new(&army_text).italics().color(
                                egui::Color32::from_rgb(180, 175, 160),
                            ));
                        }
                    }
                }
            }

            // If nothing selected
            if selection.province.is_none()
                && selection.faction.is_none()
                && selection.character.is_none()
                && selection.army.is_none()
            {
                ui.heading("Crown & Ash");
                // Era overview narrative
                if let Some(era_text) = narrative.era_summary() {
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new(era_text).italics().color(
                        egui::Color32::from_rgb(200, 190, 160),
                    ));
                    ui.add_space(4.0);
                } else {
                    ui.label("Click a province on the map to view details.");
                }
                ui.separator();
                ui.label("Keyboard:");
                ui.label("  WASD / Arrows — Pan camera");
                ui.label("  Scroll — Zoom in/out");
                ui.label("  Left click — Select province");
            }
        });
}

// ---------------------------------------------------------------------------
// Event feed — bottom panel, scrolling log of narrative events.
// ---------------------------------------------------------------------------

pub fn event_feed(
    mut contexts: EguiContexts,
    game_state: Res<ClientGameState>,
    narrative: Res<NarrativeState>,
) {
    let ctx = contexts.ctx_mut();

    egui::TopBottomPanel::bottom("crown_ash_event_feed")
        .default_height(150.0)
        .min_height(80.0)
        .show(ctx, |ui| {
            ui.heading("Chronicle");
            ui.separator();

            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    // Prefer narrative prose if available, fall back to raw format
                    if !narrative.event_narratives.is_empty() {
                        let start = narrative.event_narratives.len().saturating_sub(100);
                        let mut last_turn_shown: u32 = 0;
                        for entry in &narrative.event_narratives[start..] {
                            // Insert turn summary header when turn changes
                            if entry.turn != last_turn_shown && entry.turn > 0 {
                                if last_turn_shown > 0 {
                                    ui.add_space(6.0);
                                }
                                if let Some(summary) = narrative.turn_summary(entry.turn) {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(220, 190, 120),
                                        egui::RichText::new(summary).strong().italics(),
                                    );
                                } else {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(160, 160, 160),
                                        egui::RichText::new(format!("— Turn {} —", entry.turn)).italics(),
                                    );
                                }
                                ui.add_space(2.0);
                                last_turn_shown = entry.turn;
                            }

                            let color = match entry.importance {
                                NarrativeImportance::Epic => egui::Color32::from_rgb(255, 200, 50),
                                NarrativeImportance::Notable => egui::Color32::from_rgb(180, 200, 255),
                                NarrativeImportance::Minor => egui::Color32::from_rgb(180, 180, 180),
                            };
                            ui.colored_label(color, &entry.prose);
                            ui.add_space(2.0);
                        }
                    } else if !game_state.events.is_empty() {
                        // Fallback: raw event formatting (before narrative engine runs)
                        let start = game_state.events.len().saturating_sub(100);
                        for event in &game_state.events[start..] {
                            ui.label(format_event(event));
                        }
                    } else {
                        ui.label("The chronicle awaits its first entry...");
                    }
                });
        });
}

// ---------------------------------------------------------------------------
// Event formatting
// ---------------------------------------------------------------------------

fn format_event(event: &GameEvent) -> String {
    match event {
        GameEvent::Battle(result) => {
            format!(
                "[Battle] Province {} — {} casualties",
                result.province,
                result.attacker_casualties + result.defender_casualties
            )
        }
        GameEvent::ProvinceConquered { province, old_controller, new_controller, turn } => {
            format!("[Turn {}] Province {} conquered: faction {} -> {}", turn, province, old_controller, new_controller)
        }
        GameEvent::WarDeclared { attacker, defender, casus_belli, turn } => {
            format!("[Turn {}] War declared: faction {} vs {} ({})", turn, attacker, defender, casus_belli)
        }
        GameEvent::TreatySigned { faction_a, faction_b, treaty_type, turn } => {
            format!("[Turn {}] Treaty signed: {} & {} ({})", turn, faction_a, faction_b, treaty_type)
        }
        GameEvent::CharacterDied { character_name, cause, turn, .. } => {
            format!("[Turn {}] {} died ({:?})", turn, character_name, cause)
        }
        GameEvent::CharacterBorn { character_name, turn, .. } => {
            format!("[Turn {}] {} was born", turn, character_name)
        }
        GameEvent::SuccessionCrisis { faction, dead_ruler, realm_split, turn, .. } => {
            let split = if *realm_split { " (realm split!)" } else { "" };
            format!("[Turn {}] Succession crisis in faction {} after ruler #{} died{}", turn, faction, dead_ruler, split)
        }
        GameEvent::PlagueOutbreak { province, population_lost, turn, .. } => {
            format!("[Turn {}] Plague in province {} — {} died", turn, province, population_lost)
        }
        GameEvent::Famine { province, turn, .. } => {
            format!("[Turn {}] Famine in province {}", turn, province)
        }
        GameEvent::Harvest { province, turn, .. } => {
            format!("[Turn {}] Bountiful harvest in province {}", turn, province)
        }
        GameEvent::Rebellion { province, rebels, turn } => {
            format!("[Turn {}] Rebellion in province {}! {} rebels", turn, province, rebels)
        }
        GameEvent::PlayerJoined { wallet, faction, turn } => {
            format!("[Turn {}] Player joined: {} as faction {}", turn, wallet, faction)
        }
        GameEvent::ConstructionComplete { province, improvement, turn } => {
            format!("[Turn {}] {} completed in province {}", turn, improvement, province)
        }
        GameEvent::FactionEliminated { faction, turn } => {
            format!("[Turn {}] Faction {} eliminated!", turn, faction)
        }
        GameEvent::RealmSplit { original_faction, new_faction, provinces_lost, turn, .. } => {
            format!("[Turn {}] Realm split! Faction {} lost {} provinces to new faction {}", turn, original_faction, provinces_lost, new_faction)
        }
        GameEvent::PlotLaunched { plot_type, turn, .. } => {
            format!("[Turn {}] Plot launched: {}", turn, plot_type)
        }
        GameEvent::PlotSucceeded { instigator_name, target_name, plot_type, turn } => {
            format!("[Turn {}] {} succeeded in {} against {}", turn, instigator_name, plot_type, target_name)
        }
        GameEvent::PlotDiscovered { instigator_name, discovered_by, turn, .. } => {
            format!("[Turn {}] Plot by {} discovered by {}", turn, instigator_name, discovered_by)
        }
        GameEvent::PlotFoiled { instigator_name, target_name, turn } => {
            format!("[Turn {}] Plot by {} against {} foiled", turn, instigator_name, target_name)
        }
        GameEvent::TradeRouteEstablished { from, to, goods, turn } => {
            format!("[Turn {}] Trade route: {} -> {} ({})", turn, from, to, goods)
        }
        GameEvent::TradeRouteDisrupted { from, to, reason, turn } => {
            format!("[Turn {}] Trade route {} -> {} disrupted: {}", turn, from, to, reason)
        }
        GameEvent::CharacterTombstoned { character_name, turn, .. } => {
            format!("[Turn {}] {} passed into memory", turn, character_name)
        }
        GameEvent::ArmyAutoDisbanded { army_id, faction, province, turn, .. } => {
            format!("[Turn {}] Army #{} (faction {}) disbanded at province {}", turn, army_id, faction, province)
        }
        GameEvent::ReligiousConversion { province, old_religion, new_religion, turn } => {
            format!("[Turn {}] Province {} converted from {} to {}", turn, province, old_religion, new_religion)
        }
        GameEvent::Heresy { faction, province, severity, turn } => {
            format!("[Turn {}] Heresy in faction {} at province {} (severity {})", turn, faction, province, severity)
        }
        GameEvent::Miracle { province, prosperity_gain, turn } => {
            format!("[Turn {}] Miracle at province {}! +{} prosperity", turn, province, prosperity_gain)
        }
        GameEvent::SiegeStarted { province, attacker_faction, turns_required, turn, .. } => {
            format!("[Turn {}] Siege begun at province {} by faction {} ({} turns)", turn, province, attacker_faction, turns_required)
        }
        GameEvent::SiegeCompleted { province, old_controller, new_controller, turns_lasted, turn, .. } => {
            format!("[Turn {}] Siege of province {} complete after {} turns: {} -> {}", turn, province, turns_lasted, old_controller, new_controller)
        }
        GameEvent::Friendship { character_a, character_b, turn } => {
            format!("[Turn {}] Characters {} and {} became friends", turn, character_a, character_b)
        }
        GameEvent::Rivalry { character_a, character_b, turn } => {
            format!("[Turn {}] Characters {} and {} became rivals", turn, character_a, character_b)
        }
        GameEvent::MarriageAlliance { faction_a, faction_b, turn, .. } => {
            format!("[Turn {}] Marriage alliance formed between factions {} and {}", turn, faction_a, faction_b)
        }
    }
}

// ---------------------------------------------------------------------------
// Minimap — small overview of all 25 provinces with faction colours.
// ---------------------------------------------------------------------------

/// Province positions on the XZ plane (matching map_render::PROVINCE_POSITIONS).
const MINIMAP_POSITIONS: [(f32, f32); 25] = [
    (-4.5, -9.0), (-1.5, -9.0), (1.5, -9.0), (4.5, -9.0),
    (0.0, 3.0), (3.0, 3.0), (6.0, 3.0),
    (-1.5, -3.0), (1.5, -3.0), (1.5, 0.0), (4.5, 0.0),
    (-4.5, 0.0), (-1.5, 3.0), (0.0, 6.0),
    (7.5, 0.0), (9.0, 3.0), (10.5, 0.0), (9.0, 6.0),
    (-7.5, -6.0), (-7.5, -3.0), (-4.5, -3.0),
    (7.5, -9.0), (7.5, -6.0), (10.5, -3.0), (10.5, -6.0),
];

/// Default faction colours [R,G,B].
const FACTION_COLORS: [[u8; 3]; 7] = [
    [200, 50, 50],    // 0: Ashen Crown — crimson
    [50, 50, 200],    // 1: Vale Princes — blue
    [200, 180, 30],   // 2: Ember Church — gold
    [40, 180, 180],   // 3: Salt League — teal
    [180, 180, 220],  // 4: Frost Marches — pale ice-blue
    [180, 80, 40],    // 5: Red Steppe — rust
    [90, 40, 130],    // 6: Black Abbey — dark purple
];

/// Adjacency pairs for minimap lines.
const MINIMAP_ADJACENCY: [(usize, usize); 40] = [
    (0,1),(1,2),(2,3),(0,18),(0,20),(1,7),(1,8),(2,8),(3,21),(3,22),
    (4,9),(4,12),(5,9),(5,10),(5,6),(6,14),(7,8),(7,11),(7,20),(8,9),
    (9,10),(9,12),(10,14),(10,5),(11,12),(11,19),(11,20),(12,13),(13,17),
    (14,15),(14,16),(15,16),(15,17),(18,19),(19,20),(21,22),(22,23),(23,24),
    (22,24),(21,3),
];

pub fn minimap(
    mut contexts: EguiContexts,
    game_state: Res<ClientGameState>,
    narrative: Res<NarrativeState>,
    mut selection: ResMut<Selection>,
) {
    let ctx = contexts.ctx_mut();

    egui::Window::new("Minimap")
        .anchor(egui::Align2::RIGHT_BOTTOM, [-10.0, -170.0])
        .default_width(200.0)
        .default_height(200.0)
        .resizable(false)
        .collapsible(true)
        .show(ctx, |ui| {
            let (response, painter) = ui.allocate_painter(
                egui::vec2(200.0, 200.0),
                egui::Sense::click(),
            );
            let rect = response.rect;

            // Map world coords to minimap pixel coords.
            // World X range: roughly -7.5 to 10.5 => 18 units
            // World Z range: roughly -9 to 6 => 15 units
            let world_min_x = -8.5_f32;
            let world_max_x = 11.5_f32;
            let world_min_z = -10.0_f32;
            let world_max_z = 7.0_f32;
            let world_w = world_max_x - world_min_x;
            let world_h = world_max_z - world_min_z;

            let to_screen = |wx: f32, wz: f32| -> egui::Pos2 {
                let nx = (wx - world_min_x) / world_w;
                let ny = (wz - world_min_z) / world_h;
                egui::pos2(
                    rect.min.x + nx * rect.width(),
                    rect.min.y + ny * rect.height(),
                )
            };

            // Background
            painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(30, 30, 40));

            // Draw adjacency lines
            for &(a, b) in &MINIMAP_ADJACENCY {
                if a < MINIMAP_POSITIONS.len() && b < MINIMAP_POSITIONS.len() {
                    let (ax, az) = MINIMAP_POSITIONS[a];
                    let (bx, bz) = MINIMAP_POSITIONS[b];
                    painter.line_segment(
                        [to_screen(ax, az), to_screen(bx, bz)],
                        egui::Stroke::new(0.5, egui::Color32::from_rgb(60, 60, 70)),
                    );
                }
            }

            // Draw province dots
            let world_data = game_state.world.as_ref();

            for (i, &(px, pz)) in MINIMAP_POSITIONS.iter().enumerate() {
                let center = to_screen(px, pz);
                let radius = 5.0;

                // Resolve colour from game state or default
                let color = if let Some(world) = world_data {
                    if let Some(prov) = world.provinces.iter().find(|p| p.id == i as u16) {
                        let ctrl = prov.controller as usize;
                        if ctrl < world.factions.len() {
                            let rgb = world.factions[ctrl].color_rgb;
                            egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
                        } else if ctrl < FACTION_COLORS.len() {
                            let c = FACTION_COLORS[ctrl];
                            egui::Color32::from_rgb(c[0], c[1], c[2])
                        } else {
                            egui::Color32::GRAY
                        }
                    } else {
                        egui::Color32::GRAY
                    }
                } else if i / 4 < FACTION_COLORS.len() {
                    let c = FACTION_COLORS[i / 4];
                    egui::Color32::from_rgb(c[0], c[1], c[2])
                } else {
                    egui::Color32::GRAY
                };

                painter.circle_filled(center, radius, color);

                // Highlight selected province
                if selection.province == Some(i as u16) {
                    painter.circle_stroke(
                        center,
                        radius + 2.0,
                        egui::Stroke::new(2.0, egui::Color32::WHITE),
                    );
                }
            }

            // Draw war indicator lines between warring factions (red dashed).
            if let Some(world) = world_data {
                // Collect war pairs from realms.
                let mut war_pairs: Vec<(u8, u8)> = Vec::new();
                for realm in &world.realms {
                    for &enemy in &realm.at_war_with {
                        let a = realm.faction.min(enemy);
                        let b = realm.faction.max(enemy);
                        if !war_pairs.contains(&(a, b)) {
                            war_pairs.push((a, b));
                        }
                    }
                }

                // For each war pair, draw a red line between faction capitals.
                for (fa, fb) in &war_pairs {
                    // Find a representative province for each faction (first controlled).
                    let prov_a = world.provinces.iter().find(|p| p.controller == *fa);
                    let prov_b = world.provinces.iter().find(|p| p.controller == *fb);
                    if let (Some(pa), Some(pb)) = (prov_a, prov_b) {
                        let idx_a = pa.id as usize;
                        let idx_b = pb.id as usize;
                        if idx_a < MINIMAP_POSITIONS.len() && idx_b < MINIMAP_POSITIONS.len() {
                            let (ax, az) = MINIMAP_POSITIONS[idx_a];
                            let (bx, bz) = MINIMAP_POSITIONS[idx_b];
                            let sa = to_screen(ax, az);
                            let sb = to_screen(bx, bz);

                            // Draw dashed red line (3 dashes along the segment).
                            let war_color = egui::Color32::from_rgba_premultiplied(220, 50, 50, 180);
                            let num_dashes = 5;
                            for d in 0..num_dashes {
                                let t0 = d as f32 / num_dashes as f32;
                                let t1 = (d as f32 + 0.6) / num_dashes as f32;
                                let p0 = egui::pos2(
                                    sa.x + (sb.x - sa.x) * t0,
                                    sa.y + (sb.y - sa.y) * t0,
                                );
                                let p1 = egui::pos2(
                                    sa.x + (sb.x - sa.x) * t1.min(1.0),
                                    sa.y + (sb.y - sa.y) * t1.min(1.0),
                                );
                                painter.line_segment(
                                    [p0, p1],
                                    egui::Stroke::new(1.5, war_color),
                                );
                            }
                        }
                    }
                }
            }

            // Handle click — select nearest province on minimap
            if response.clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let mut best: Option<(u16, f32)> = None;
                    for (i, &(px, pz)) in MINIMAP_POSITIONS.iter().enumerate() {
                        let screen_pos = to_screen(px, pz);
                        let dist = pos.distance(screen_pos);
                        if dist < 15.0 {
                            if best.map_or(true, |(_, bd)| dist < bd) {
                                best = Some((i as u16, dist));
                            }
                        }
                    }
                    if let Some((pid, _)) = best {
                        selection.province = Some(pid);
                    }
                }
            }

            // Hover tooltip — show brief province narrative
            if response.hovered() {
                if let Some(hover_pos) = ui.input(|i| i.pointer.hover_pos()) {
                    // Find nearest province to cursor
                    let mut best_hover: Option<(u16, f32)> = None;
                    for (i, &(px, pz)) in MINIMAP_POSITIONS.iter().enumerate() {
                        let screen_pos = to_screen(px, pz);
                        let dist = hover_pos.distance(screen_pos);
                        if dist < 12.0 {
                            if best_hover.map_or(true, |(_, bd)| dist < bd) {
                                best_hover = Some((i as u16, dist));
                            }
                        }
                    }

                    if let Some((pid, _)) = best_hover {
                        if let Some(world) = world_data {
                            let prov_name = world.provinces.iter()
                                .find(|p| p.id == pid)
                                .map(|p| p.name.as_str())
                                .unwrap_or("Unknown");
                            let controller = world.provinces.iter()
                                .find(|p| p.id == pid)
                                .map(|p| {
                                    world.factions.iter()
                                        .find(|f| f.id == p.controller)
                                        .map(|f| f.name.as_str())
                                        .unwrap_or("Uncontrolled")
                                })
                                .unwrap_or("Unknown");

                            // Build tooltip with narrative snippet
                            let mut tip = format!("{} ({})", prov_name, controller);
                            if let Some(history) = narrative.province_history(pid) {
                                // Take the last sentence of the history for a brief hint
                                let last_sentence = history.rsplit(". ")
                                    .next()
                                    .unwrap_or(history);
                                if !last_sentence.is_empty() && last_sentence.len() < 120 {
                                    tip.push_str("\n");
                                    tip.push_str(last_sentence);
                                }
                            }

                            response.clone().on_hover_text(tip);
                        }
                    }
                }
            }
        });
}

// ---------------------------------------------------------------------------
// Faction list — left panel showing all factions at a glance.
// ---------------------------------------------------------------------------

pub fn faction_list(
    mut contexts: EguiContexts,
    game_state: Res<ClientGameState>,
    mut selection: ResMut<Selection>,
) {
    let ctx = contexts.ctx_mut();

    egui::SidePanel::left("crown_ash_factions")
        .default_width(220.0)
        .min_width(180.0)
        .show(ctx, |ui| {
            ui.heading("Factions");
            ui.separator();

            let Some(ref world) = game_state.world else {
                ui.label("Waiting for world...");
                return;
            };

            for faction in &world.factions {
                let prov_count = world.provinces.iter()
                    .filter(|p| p.controller == faction.id)
                    .count();
                let army_count = world.armies.iter()
                    .filter(|a| a.owner_faction == faction.id)
                    .count();
                let pop: u64 = world.provinces.iter()
                    .filter(|p| p.controller == faction.id)
                    .map(|p| p.population as u64)
                    .sum();

                let is_selected = selection.faction == Some(faction.id);
                let status = if faction.alive { "" } else { " [DEAD]" };

                // Faction colour chip.
                let [r, g, b] = faction.color_rgb;
                let chip_color = egui::Color32::from_rgb(r, g, b);

                ui.horizontal(|ui| {
                    // Colour dot.
                    let (rect, _) = ui.allocate_exact_size(
                        egui::vec2(12.0, 12.0),
                        egui::Sense::hover(),
                    );
                    ui.painter().circle_filled(rect.center(), 5.0, chip_color);

                    // Clickable faction name.
                    let label = if is_selected {
                        egui::RichText::new(format!("{}{}", faction.name, status))
                            .strong()
                            .color(egui::Color32::WHITE)
                    } else if faction.alive {
                        egui::RichText::new(format!("{}{}", faction.name, status))
                    } else {
                        egui::RichText::new(format!("{}{}", faction.name, status))
                            .strikethrough()
                            .color(egui::Color32::GRAY)
                    };

                    if ui.add(egui::Label::new(label).sense(egui::Sense::click())).clicked() {
                        selection.faction = Some(faction.id);
                        // Also select the faction's capital (first province) if any.
                        if let Some(prov) = world.provinces.iter().find(|p| p.controller == faction.id) {
                            selection.province = Some(prov.id);
                        }
                    }
                });

                // Summary stats indented below.
                ui.indent(faction.id as usize, |ui| {
                    ui.label(format!("{} prov / {} army / {}", prov_count, army_count, format_population(pop)));

                    // Show wars.
                    if let Some(realm) = world.realms.iter().find(|r| r.faction == faction.id) {
                        if !realm.at_war_with.is_empty() {
                            let enemies: Vec<&str> = realm.at_war_with.iter()
                                .filter_map(|&eid| world.factions.iter()
                                    .find(|f| f.id == eid)
                                    .map(|f| f.name.as_str()))
                                .collect();
                            ui.colored_label(
                                egui::Color32::from_rgb(220, 80, 80),
                                format!("At war: {}", enemies.join(", ")),
                            );
                        }
                    }
                });

                ui.add_space(4.0);
            }

            // Turn info at the bottom.
            ui.separator();
            ui.label(format!("Turn: {}", world.meta.turn));
            ui.label("Turns advance with each block.");
        });
}

// ---------------------------------------------------------------------------
// Join Game dialog — shown when player hasn't joined yet.
// ---------------------------------------------------------------------------

pub fn join_dialog(
    mut contexts: EguiContexts,
    game_state: Res<ClientGameState>,
    config: Res<CrownAshConfig>,
    mut join_state: ResMut<JoinState>,
) {
    // Drain join-game async result.
    let pending = Arc::clone(&join_state.pending);
    if let Ok(mut lock) = pending.try_lock() {
        if let Some(result) = lock.take() {
            drop(lock);
            join_state.joining = false;
            match result {
                Ok(msg) => {
                    join_state.joined = true;
                    join_state.join_result = Some(msg);
                }
                Err(msg) => {
                    // If the wallet already controls a faction, the player is
                    // already in the game — dismiss the join dialog and proceed.
                    if msg.contains("already controls") {
                        join_state.joined = true;
                        join_state.join_result = Some("Rejoined — welcome back!".to_string());
                    } else {
                        join_state.join_result = Some(msg);
                    }
                }
            }
        }
    }

    // Drain device-login async result.
    let dl_pending = Arc::clone(&join_state.device_login_pending);
    if let Ok(mut lock) = dl_pending.try_lock() {
        if let Some(result) = lock.take() {
            drop(lock);
            match result {
                Ok(phase) => {
                    if let DeviceLoginPhase::Complete { ref wallet_address } = phase {
                        join_state.wallet_input = wallet_address.clone();
                    }
                    join_state.device_login = phase;
                }
                Err(msg) => {
                    join_state.device_login = DeviceLoginPhase::Error(msg);
                }
            }
        }
    }

    // Don't show dialog if already joined.
    if join_state.joined {
        return;
    }

    let ctx = contexts.ctx_mut();

    egui::Window::new("Join Crown & Ash")
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .default_width(420.0)
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| {
            ui.heading("Enter the Realm");
            ui.separator();

            // ── OAuth2 Device Login ──
            if !join_state.manual_mode {
                match &join_state.device_login {
                    DeviceLoginPhase::Idle => {
                        ui.add_space(8.0);
                        if ui.add(egui::Button::new(
                            egui::RichText::new("Login with Quillon Wallet")
                                .size(16.0)
                                .strong()
                        ).min_size(egui::vec2(380.0, 36.0))).clicked() {
                            // Start device-login request.
                            join_state.device_login = DeviceLoginPhase::Requesting;
                            let base_url = config.server_url.clone();
                            let slot = Arc::clone(&join_state.device_login_pending);

                            std::thread::Builder::new()
                                .name("crown-ash-login".into())
                                .spawn(move || {
                                    let rt = tokio::runtime::Builder::new_current_thread()
                                        .enable_all()
                                        .build()
                                        .expect("tokio runtime for device login");
                                    rt.block_on(async move {
                                        let client = reqwest::Client::new();
                                        let url = format!("{}/api/v1/miner/device-login", base_url);
                                        let result = request_device_login(&client, &url, slot.clone()).await;
                                        if let Ok(mut lock) = slot.lock() {
                                            *lock = Some(result);
                                        }
                                    });
                                })
                                .ok();
                        }
                        ui.add_space(4.0);
                        ui.label("Sign in with your browser — no password entered here.");
                    }

                    DeviceLoginPhase::Requesting => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Requesting login code...");
                        });
                    }

                    DeviceLoginPhase::WaitingForApproval { device_code, verification_url } => {
                        ui.add_space(4.0);
                        ui.label("Open this URL in your browser:");
                        ui.add_space(4.0);

                        // Clickable link.
                        let link = egui::RichText::new(verification_url.as_str())
                            .color(egui::Color32::from_rgb(100, 180, 255))
                            .underline();
                        if ui.add(egui::Label::new(link).sense(egui::Sense::click())).clicked() {
                            let _ = open::that(verification_url);
                        }

                        ui.add_space(8.0);
                        ui.label(format!("Code: {}", device_code));
                        ui.add_space(8.0);

                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Waiting for approval...");
                        });

                        // Open browser button.
                        if ui.button("Open Browser").clicked() {
                            let _ = open::that(verification_url);
                        }
                    }

                    DeviceLoginPhase::Complete { wallet_address } => {
                        ui.colored_label(
                            egui::Color32::GREEN,
                            format!("Logged in: {}", &wallet_address[..wallet_address.len().min(20)]),
                        );
                    }

                    DeviceLoginPhase::Error(msg) => {
                        ui.colored_label(egui::Color32::RED, format!("Login error: {}", msg));
                        if ui.button("Retry").clicked() {
                            join_state.device_login = DeviceLoginPhase::Idle;
                        }
                    }
                }

                ui.add_space(4.0);
                if ui.small_button("Enter wallet manually instead").clicked() {
                    join_state.manual_mode = true;
                }
            } else {
                // ── Manual wallet input (fallback) ──
                ui.label("Wallet address:");
                ui.text_edit_singleline(&mut join_state.wallet_input);
                ui.add_space(4.0);
                if ui.small_button("Use Quillon Wallet login instead").clicked() {
                    join_state.manual_mode = false;
                }
            }

            ui.add_space(8.0);

            // ── Faction picker ──
            let Some(ref world) = game_state.world else {
                ui.label("Waiting for world data...");
                return;
            };

            let wallet_ready = !join_state.wallet_input.is_empty()
                || matches!(join_state.device_login, DeviceLoginPhase::Complete { .. });

            if wallet_ready {
                ui.separator();
                ui.label("Choose your faction:");
                ui.add_space(4.0);

                for faction in &world.factions {
                    if !faction.alive {
                        continue;
                    }
                    let claimed = faction.player_wallet.is_some();
                    let is_selected = join_state.selected_faction == faction.id;

                    let [r, g, b] = faction.color_rgb;
                    let chip = egui::Color32::from_rgb(r, g, b);

                    ui.horizontal(|ui| {
                        let (rect, _) = ui.allocate_exact_size(
                            egui::vec2(14.0, 14.0),
                            egui::Sense::hover(),
                        );
                        ui.painter().circle_filled(rect.center(), 6.0, chip);

                        let label_text = if claimed {
                            format!("{} (taken)", faction.name)
                        } else {
                            faction.name.clone()
                        };

                        let rich = if is_selected && !claimed {
                            egui::RichText::new(label_text).strong().color(egui::Color32::WHITE)
                        } else if claimed {
                            egui::RichText::new(label_text).color(egui::Color32::GRAY).strikethrough()
                        } else {
                            egui::RichText::new(label_text)
                        };

                        let resp = ui.add(egui::Label::new(rich).sense(egui::Sense::click()));
                        if resp.clicked() && !claimed {
                            join_state.selected_faction = faction.id;
                        }
                    });

                    if is_selected && !claimed {
                        let prov_count = world.provinces.iter()
                            .filter(|p| p.controller == faction.id)
                            .count();
                        ui.indent(faction.id as usize + 100, |ui| {
                            ui.label(format!("  {} provinces, {:?} culture, {:?} religion",
                                prov_count, faction.culture, faction.religion));
                        });
                    }
                }

                ui.add_space(12.0);
                ui.separator();

                // Join button.
                let can_join = !join_state.wallet_input.is_empty() && !join_state.joining;
                ui.add_enabled_ui(can_join, |ui| {
                    if ui.button(egui::RichText::new("Join Game").size(15.0).strong()).clicked() {
                        let url = format!("{}/api/v1/crown-ash/join", config.server_url);
                        let body = serde_json::json!({
                            "wallet": join_state.wallet_input,
                            "faction": join_state.selected_faction
                        });
                        let slot = Arc::clone(&join_state.pending);
                        join_state.joining = true;
                        join_state.join_result = None;

                        std::thread::Builder::new()
                            .name("crown-ash-join".into())
                            .spawn(move || {
                                let rt = tokio::runtime::Builder::new_current_thread()
                                    .enable_all()
                                    .build()
                                    .expect("tokio runtime for join");
                                rt.block_on(async move {
                                    let client = reqwest::Client::new();
                                    let result = match client.post(&url).json(&body).send().await {
                                        Ok(resp) => {
                                            if resp.status().is_success() {
                                                Ok("Welcome to Crown & Ash!".to_string())
                                            } else {
                                                let body = resp.text().await.unwrap_or_default();
                                                Err(format!("Join failed: {}", body))
                                            }
                                        }
                                        Err(e) => Err(format!("Network error: {}", e)),
                                    };
                                    if let Ok(mut lock) = slot.lock() {
                                        *lock = Some(result);
                                    }
                                });
                            })
                            .ok();
                    }
                });
            }

            // Observer mode — always available.
            if ui.button("Watch as Observer").clicked() {
                join_state.joined = true;
                join_state.join_result = Some("Observing".to_string());
            }

            if join_state.joining {
                ui.spinner();
            }
            if let Some(ref msg) = join_state.join_result {
                ui.colored_label(
                    if join_state.joined { egui::Color32::GREEN } else { egui::Color32::RED },
                    msg,
                );
            }
        });
}

// ---------------------------------------------------------------------------
// OAuth2 device-login helper functions
// ---------------------------------------------------------------------------

/// Request a device login code, then poll until the user approves in their browser.
///
/// Flow:
/// 1. POST /api/v1/miner/device-login → get device_code + verification_url
/// 2. Open browser to verification_url
/// 3. Write WaitingForApproval phase to slot (UI shows URL + code)
/// 4. Poll GET /api/v1/miner/device-login/{code} every 3s
/// 5. When status=complete → return Complete { wallet_address }
async fn request_device_login(
    client: &reqwest::Client,
    url: &str,
    slot: Arc<Mutex<Option<Result<DeviceLoginPhase, String>>>>,
) -> Result<DeviceLoginPhase, String> {
    // Step 1: Request device code.
    let resp = client.post(url)
        .send()
        .await
        .map_err(|e| format!("Network error: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("Server error: {}", resp.status()));
    }

    let val: serde_json::Value = resp.json()
        .await
        .map_err(|e| format!("JSON error: {e}"))?;

    let device_code = val.get("device_code")
        .or_else(|| val.get("data").and_then(|d| d.get("device_code")))
        .and_then(|v| v.as_str())
        .ok_or("Missing device_code in response")?
        .to_string();

    let verification_url = val.get("verification_url")
        .or_else(|| val.get("data").and_then(|d| d.get("verification_url")))
        .and_then(|v| v.as_str())
        .unwrap_or("https://quillon.xyz/miner-login")
        .to_string();

    let verification_url = if verification_url.contains('?') {
        verification_url
    } else {
        format!("{}?code={}", verification_url, device_code)
    };

    // Step 2: Open browser automatically.
    let _ = open::that(&verification_url);

    // Step 3: Push WaitingForApproval phase to UI.
    let waiting_phase = DeviceLoginPhase::WaitingForApproval {
        device_code: device_code.clone(),
        verification_url: verification_url.clone(),
    };
    if let Ok(mut lock) = slot.lock() {
        *lock = Some(Ok(waiting_phase));
    }

    // Step 4: Poll for completion (up to 10 minutes).
    let poll_url = format!("{}/{}", url, device_code);
    for _ in 0..200 {
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;

        let resp = match client.get(&poll_url).send().await {
            Ok(r) => r,
            Err(_) => continue,
        };

        if !resp.status().is_success() {
            continue;
        }

        let val: serde_json::Value = match resp.json().await {
            Ok(v) => v,
            Err(_) => continue,
        };

        let status = val.get("status")
            .or_else(|| val.get("data").and_then(|d| d.get("status")))
            .and_then(|v| v.as_str())
            .unwrap_or("pending");

        if status == "complete" || status == "approved" {
            let wallet = val.get("wallet_address")
                .or_else(|| val.get("data").and_then(|d| d.get("wallet_address")))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            if !wallet.is_empty() {
                return Ok(DeviceLoginPhase::Complete { wallet_address: wallet });
            }
        }
    }

    Err("Device login expired (10 min timeout)".into())
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------

pub fn keyboard_shortcuts(
    keys: Res<ButtonInput<KeyCode>>,
    game_state: Res<ClientGameState>,
    mut selection: ResMut<Selection>,
) {
    // ESC — clear all selection.
    if keys.just_pressed(KeyCode::Escape) {
        selection.province = None;
        selection.faction = None;
        selection.character = None;
        selection.army = None;
    }

    // Tab — cycle to next alive faction and select its first province.
    if keys.just_pressed(KeyCode::Tab) {
        let Some(ref world) = game_state.world else {
            return;
        };

        let alive: Vec<u8> = world.factions.iter()
            .filter(|f| f.alive)
            .map(|f| f.id)
            .collect();

        if alive.is_empty() {
            return;
        }

        let current = selection.faction.unwrap_or(255);
        let next = alive.iter()
            .find(|&&id| id > current)
            .or_else(|| alive.first())
            .copied()
            .unwrap_or(0);

        selection.faction = Some(next);
        if let Some(prov) = world.provinces.iter().find(|p| p.controller == next) {
            selection.province = Some(prov.id);
        }
        selection.army = None;
    }

    // Number keys 1-7 — direct faction select.
    let number_keys = [
        (KeyCode::Digit1, 0u8),
        (KeyCode::Digit2, 1),
        (KeyCode::Digit3, 2),
        (KeyCode::Digit4, 3),
        (KeyCode::Digit5, 4),
        (KeyCode::Digit6, 5),
        (KeyCode::Digit7, 6),
    ];

    for (key, faction_id) in number_keys {
        if keys.just_pressed(key) {
            let Some(ref world) = game_state.world else {
                return;
            };
            if world.factions.iter().any(|f| f.id == faction_id && f.alive) {
                selection.faction = Some(faction_id);
                if let Some(prov) = world.provinces.iter().find(|p| p.controller == faction_id) {
                    selection.province = Some(prov.id);
                }
                selection.army = None;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dialog speech bubbles — floating overlay for Tier 2/3 LLM-generated text
// ---------------------------------------------------------------------------

/// System: renders floating speech bubbles for LLM-generated dialog and
/// narrative text.  Bubbles auto-dismiss after a timer (8s for dialog,
/// 12s for epic narrative).  Stacked vertically from the top-right.
pub fn dialog_bubbles(
    mut contexts: EguiContexts,
    mut dialog: ResMut<DialogState>,
    time: Res<Time>,
) {
    // Tick timers and remove expired bubbles.
    dialog.tick(time.delta_secs());

    if dialog.bubbles.is_empty() {
        return;
    }

    let ctx = contexts.ctx_mut();

    for (i, bubble) in dialog.bubbles.iter().enumerate() {
        // Stack bubbles vertically from the top of the viewport, offset right.
        let y_offset = 50.0 + i as f32 * 120.0;

        // Fade out in the last 1.5 seconds.
        let alpha = if bubble.timer < 1.5 {
            (bubble.timer / 1.5).clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Tier-based styling.
        let (bg_color, border_color, speaker_color) = match bubble.tier {
            3 => (
                // Epic: dark gold background
                egui::Color32::from_rgba_unmultiplied(40, 30, 10, (220.0 * alpha) as u8),
                egui::Color32::from_rgba_unmultiplied(255, 200, 50, (200.0 * alpha) as u8),
                egui::Color32::from_rgba_unmultiplied(255, 200, 50, (255.0 * alpha) as u8),
            ),
            _ => (
                // Dialog: dark blue-grey background
                egui::Color32::from_rgba_unmultiplied(25, 30, 45, (220.0 * alpha) as u8),
                egui::Color32::from_rgba_unmultiplied(140, 170, 220, (180.0 * alpha) as u8),
                egui::Color32::from_rgba_unmultiplied(180, 200, 255, (255.0 * alpha) as u8),
            ),
        };

        let text_color = egui::Color32::from_rgba_unmultiplied(
            220, 220, 220, (255.0 * alpha) as u8,
        );

        let window_id = format!("dialog_bubble_{}", i);
        egui::Window::new("")
            .id(egui::Id::new(&window_id))
            .anchor(egui::Align2::RIGHT_TOP, [-20.0, y_offset])
            .fixed_size([320.0, 0.0])
            .title_bar(false)
            .resizable(false)
            .frame(egui::Frame::new()
                .fill(bg_color)
                .stroke(egui::Stroke::new(1.5, border_color))
                .corner_radius(8.0)
                .inner_margin(10.0))
            .show(ctx, |ui| {
                // Speaker name with a speech marker.
                let speaker_label = if bubble.tier == 3 {
                    format!("~ {} ~", bubble.speaker)
                } else {
                    format!("{} says:", bubble.speaker)
                };
                ui.label(
                    egui::RichText::new(speaker_label)
                        .strong()
                        .size(13.0)
                        .color(speaker_color),
                );
                ui.add_space(4.0);

                // Dialog text (italic for epic, normal for dialog).
                let text_rich = if bubble.tier == 3 {
                    egui::RichText::new(&bubble.text)
                        .italics()
                        .size(12.0)
                        .color(text_color)
                } else {
                    egui::RichText::new(format!("\"{}\"", bubble.text))
                        .size(12.0)
                        .color(text_color)
                };
                ui.label(text_rich);

                // Thin progress bar showing remaining time.
                let max_time = if bubble.tier == 3 { 12.0 } else { 8.0 };
                let fraction = (bubble.timer / max_time).clamp(0.0, 1.0);
                let bar = egui::ProgressBar::new(fraction)
                    .desired_width(ui.available_width());
                ui.add_space(4.0);
                ui.add(bar);
            });
    }
}

// ---------------------------------------------------------------------------
// Notification toasts — brief auto-dismiss popups for Epic/Notable events
// ---------------------------------------------------------------------------

/// System: renders notification toasts on the left side of the screen.
/// Shows brief summaries of Epic and Notable events as they happen.
pub fn notification_toasts(
    mut contexts: EguiContexts,
    mut toasts: ResMut<ToastState>,
    time: Res<Time>,
) {
    toasts.tick(time.delta_secs());

    if toasts.toasts.is_empty() {
        return;
    }

    let ctx = contexts.ctx_mut();

    for (i, toast) in toasts.toasts.iter().enumerate() {
        let y_offset = 60.0 + i as f32 * 50.0;

        let alpha = if toast.timer < 1.0 {
            (toast.timer).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let (bg, border, text_color) = match toast.importance {
            NarrativeImportance::Epic => (
                egui::Color32::from_rgba_unmultiplied(50, 35, 10, (200.0 * alpha) as u8),
                egui::Color32::from_rgba_unmultiplied(255, 200, 50, (180.0 * alpha) as u8),
                egui::Color32::from_rgba_unmultiplied(255, 220, 100, (255.0 * alpha) as u8),
            ),
            NarrativeImportance::Notable => (
                egui::Color32::from_rgba_unmultiplied(20, 30, 50, (200.0 * alpha) as u8),
                egui::Color32::from_rgba_unmultiplied(100, 150, 220, (150.0 * alpha) as u8),
                egui::Color32::from_rgba_unmultiplied(180, 200, 255, (255.0 * alpha) as u8),
            ),
            NarrativeImportance::Minor => (
                egui::Color32::from_rgba_unmultiplied(30, 30, 30, (180.0 * alpha) as u8),
                egui::Color32::from_rgba_unmultiplied(100, 100, 100, (120.0 * alpha) as u8),
                egui::Color32::from_rgba_unmultiplied(180, 180, 180, (255.0 * alpha) as u8),
            ),
        };

        let window_id = format!("toast_{}", i);
        egui::Window::new("")
            .id(egui::Id::new(&window_id))
            .anchor(egui::Align2::LEFT_TOP, [20.0, y_offset])
            .fixed_size([280.0, 0.0])
            .title_bar(false)
            .resizable(false)
            .frame(egui::Frame::new()
                .fill(bg)
                .stroke(egui::Stroke::new(1.0, border))
                .corner_radius(6.0)
                .inner_margin(8.0))
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new(&toast.text)
                        .size(11.5)
                        .color(text_color),
                );
            });
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a WorldContext from the current game state for narrative functions.
fn build_narrative_ctx(state: &ClientGameState) -> crown_ash_narrative::WorldContext {
    let Some(ref world) = state.world else {
        return crown_ash_narrative::WorldContext::default();
    };
    crown_ash_narrative::WorldContext {
        province_names: world.provinces.iter().map(|p| (p.id, p.name.clone())).collect(),
        faction_names: world.factions.iter().map(|f| (f.id, f.name.clone())).collect(),
        character_names: world.characters.iter().map(|c| (c.id, c.name.clone())).collect(),
        faction_cultures: world.factions.iter().map(|f| (f.id, format!("{:?}", f.culture))).collect(),
        army_factions: world.armies.iter().map(|a| (a.id, a.owner_faction)).collect(),
        current_turn: world.meta.turn,
    }
}

fn format_population(pop: u64) -> String {
    if pop >= 1_000_000 {
        format!("{:.1}M", pop as f64 / 1_000_000.0)
    } else if pop >= 1_000 {
        format!("{:.1}K", pop as f64 / 1_000.0)
    } else {
        format!("{}", pop)
    }
}

// ---------------------------------------------------------------------------
// Hover tooltip — compact province info at cursor position
// ---------------------------------------------------------------------------

pub fn hover_tooltip(
    mut contexts: EguiContexts,
    game_state: Res<ClientGameState>,
    selection: Res<Selection>,
) {
    let Some(pid) = selection.hovered_province else { return };
    let Some((cx, cy)) = selection.cursor_screen_pos else { return };
    let Some(ref world) = game_state.world else { return };
    let Some(prov) = world.provinces.iter().find(|p| p.id == pid) else { return };

    // Don't show tooltip if this province is already selected (detail panel covers it).
    if selection.province == Some(pid) { return; }

    let controller_name = world.factions
        .iter()
        .find(|f| f.id == prov.controller)
        .map(|f| f.name.as_str())
        .unwrap_or("Unknown");

    let ctx = contexts.ctx_mut();

    // Position tooltip offset from cursor.
    let tooltip_pos = egui::pos2(cx + 16.0, cy + 16.0);

    egui::Area::new(egui::Id::new("province_hover_tooltip"))
        .fixed_pos(tooltip_pos)
        .order(egui::Order::Tooltip)
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_max_width(200.0);
                ui.strong(&prov.name);
                ui.label(format!("{:?} | {}", prov.terrain, controller_name));
                ui.separator();

                ui.horizontal(|ui| {
                    ui.label(format!("Pop: {}", format_population(prov.population as u64)));
                    ui.separator();
                    ui.label(format!("Fort: {}", prov.fortification));
                });

                let prosperity = prov.prosperity.raw() as f32 / 10.0;
                let unrest = prov.unrest.raw() as f32 / 10.0;
                ui.horizontal(|ui| {
                    ui.label(format!("Prosperity: {:.0}%", prosperity));
                    ui.separator();
                    ui.label(format!("Unrest: {:.0}%", unrest));
                });

                if !prov.improvements.is_empty() {
                    let imps: Vec<&str> = prov.improvements.iter()
                        .map(|i| match i {
                            crown_ash_types::Improvement::Farmstead => "Farm",
                            crown_ash_types::Improvement::Mine => "Mine",
                            crown_ash_types::Improvement::Lumbercamp => "Lumber",
                            crown_ash_types::Improvement::Quarry => "Quarry",
                            crown_ash_types::Improvement::Stables => "Stables",
                            crown_ash_types::Improvement::Market => "Market",
                            crown_ash_types::Improvement::Temple => "Temple",
                            crown_ash_types::Improvement::Fortification => "Fort",
                            crown_ash_types::Improvement::University => "Uni",
                            crown_ash_types::Improvement::Port => "Port",
                            crown_ash_types::Improvement::Granary => "Granary",
                            crown_ash_types::Improvement::Hospital => "Hospital",
                        })
                        .collect();
                    ui.label(format!("Buildings: {}", imps.join(", ")));
                }

                // Show garrison if nonzero.
                let total_garrison = prov.garrison.total();
                if total_garrison > 0 {
                    ui.label(format!("Garrison: {} troops", total_garrison));
                }
            });
        });
}
