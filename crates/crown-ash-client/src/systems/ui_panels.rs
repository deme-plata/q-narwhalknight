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
        });
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
