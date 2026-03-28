//! egui-based UI panels for the Crown & Ash client.
//!
//! Three systems render the HUD:
//! - `top_bar` — turn counter, faction count, population, connection indicator
//! - `detail_panel` — province / faction / character / army details
//! - `event_feed` — scrolling narrative event log

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::resources::config::CrownAshConfig;
use crate::resources::game_state::{ClientGameState, ConnectionStatus};
use crate::resources::selection::Selection;
use crown_ash_types::{FixedPoint, GameEvent};

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
                    ConnectionStatus::Error(e) => ("Error", egui::Color32::RED),
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
                            ui.label(format!("At war with: {}", enemies.join(", ")));
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
                    ui.separator();
                    ui.label(format!("Levy: {}", army.troops.levy));
                    ui.label(format!("Men at Arms: {}", army.troops.men_at_arms));
                    ui.label(format!("Knights: {}", army.troops.knights));
                    ui.separator();
                    ui.label(format!("Morale: {}", fp_display(army.morale)));
                    ui.label(format!("Supply: {}", fp_display(army.supply)));
                    ui.label(format!("Raised turn: {}", army.raised_turn));
                }
            }

            // If nothing selected
            if selection.province.is_none()
                && selection.faction.is_none()
                && selection.character.is_none()
                && selection.army.is_none()
            {
                ui.heading("Crown & Ash");
                ui.label("Click a province on the map to view details.");
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
) {
    let ctx = contexts.ctx_mut();

    egui::TopBottomPanel::bottom("crown_ash_event_feed")
        .default_height(150.0)
        .min_height(80.0)
        .show(ctx, |ui| {
            ui.heading("Event Log");
            ui.separator();

            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    if game_state.events.is_empty() {
                        ui.label("No events yet...");
                        return;
                    }

                    // Show last 100 events (most recent at bottom).
                    let start = game_state.events.len().saturating_sub(100);
                    for event in &game_state.events[start..] {
                        ui.label(format_event(event));
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
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn format_population(pop: u64) -> String {
    if pop >= 1_000_000 {
        format!("{:.1}M", pop as f64 / 1_000_000.0)
    } else if pop >= 1_000 {
        format!("{:.1}K", pop as f64 / 1_000.0)
    } else {
        format!("{}", pop)
    }
}
