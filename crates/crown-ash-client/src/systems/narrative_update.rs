//! Narrative update system — rebuilds narrative state when new events arrive.
//!
//! Runs each frame, checks if game state has changed (new events), and
//! regenerates narrative text using the template engine (Tier 1, instant).
//!
//! This system bridges the crown-ash-narrative crate into the Bevy ECS:
//! - Converts GameEvents → narrative prose for the event feed
//! - Builds WorldContext from the live WorldSnapshot
//! - Updates character chronicles incrementally
//! - Caches province/faction history text

use bevy::prelude::*;

use crown_ash_narrative::WorldContext;
use crown_ash_narrative::cascade::CascadeEngine;
use crown_ash_narrative::chronicle;
use crown_ash_narrative::history;

use crate::resources::game_state::ClientGameState;
use crate::resources::narrative_state::{EventNarrative, NarrativeImportance, NarrativeState, ToastState};

/// Tracks how many events we've already processed to avoid reprocessing.
#[derive(Resource)]
pub struct NarrativeProgress {
    /// Number of events already converted to narrative.
    pub events_processed: usize,
    /// Last turn for which chronicles were updated.
    pub last_chronicle_turn: u32,
    /// Last turn for which histories were regenerated.
    pub last_history_turn: u32,
    /// Highest turn for which a turn summary has been generated.
    pub last_turn_summary: u32,
    /// Last turn for which war summaries were refreshed.
    pub last_war_summary_turn: u32,
    /// Last turn for which era summary was refreshed.
    pub last_era_summary_turn: u32,
}

impl Default for NarrativeProgress {
    fn default() -> Self {
        Self {
            events_processed: 0,
            last_chronicle_turn: 0,
            last_history_turn: 0,
            last_turn_summary: 0,
            last_war_summary_turn: 0,
            last_era_summary_turn: 0,
        }
    }
}

/// Build a `WorldContext` from the current game snapshot for the narrative engine.
fn build_world_context(state: &ClientGameState) -> WorldContext {
    let Some(ref world) = state.world else {
        return WorldContext::default();
    };

    let province_names = world.provinces.iter()
        .map(|p| (p.id, p.name.clone()))
        .collect();

    let faction_names = world.factions.iter()
        .map(|f| (f.id, f.name.clone()))
        .collect();

    let character_names = world.characters.iter()
        .map(|c| (c.id, c.name.clone()))
        .collect();

    let faction_cultures = world.factions.iter()
        .map(|f| (f.id, format!("{:?}", f.culture)))
        .collect();

    let army_factions = world.armies.iter()
        .map(|a| (a.id, a.owner_faction))
        .collect();

    WorldContext {
        province_names,
        faction_names,
        character_names,
        faction_cultures,
        army_factions,
        current_turn: world.meta.turn,
    }
}

/// System: Process new events through the cascade engine for narrative text.
///
/// Only processes events that haven't been seen yet (incremental).
pub fn update_event_narratives(
    game_state: Res<ClientGameState>,
    mut narrative: ResMut<NarrativeState>,
    mut progress: ResMut<NarrativeProgress>,
    mut toasts: ResMut<ToastState>,
) {
    let total_events = game_state.events.len();
    if total_events <= progress.events_processed {
        return; // No new events
    }

    let ctx = build_world_context(&game_state);
    let cascade = CascadeEngine::new();

    // Process only new events
    for event in &game_state.events[progress.events_processed..] {
        let result = cascade.process_event(event, &ctx);
        let entry = EventNarrative::from(&result);

        // Push notification toast for Epic and Notable events
        if matches!(entry.importance, NarrativeImportance::Epic | NarrativeImportance::Notable) {
            toasts.push(entry.summary.clone(), entry.importance, entry.turn);
        }

        narrative.event_narratives.push(entry);
    }

    // Cap at 500 narratives (matching event cap)
    if narrative.event_narratives.len() > 500 {
        let excess = narrative.event_narratives.len() - 500;
        narrative.event_narratives.drain(..excess);
    }

    progress.events_processed = total_events;
    narrative.dirty = true;
}

/// System: Update character chronicles when new events arrive.
///
/// Runs less frequently than event narratives (only on turn change).
pub fn update_chronicles(
    game_state: Res<ClientGameState>,
    mut narrative: ResMut<NarrativeState>,
    mut progress: ResMut<NarrativeProgress>,
) {
    let Some(ref world) = game_state.world else { return };
    let current_turn = world.meta.turn;

    if current_turn <= progress.last_chronicle_turn {
        return; // Same turn, no update needed
    }

    let ctx = build_world_context(&game_state);

    // Update chronicles with all events
    chronicle::update_chronicles(&mut narrative.chronicles_vec, &game_state.events, &ctx);

    progress.last_chronicle_turn = current_turn;
}

/// System: Regenerate province and faction history text periodically.
///
/// Runs less frequently — only every 5 turns or when dirty.
pub fn update_histories(
    game_state: Res<ClientGameState>,
    mut narrative: ResMut<NarrativeState>,
    mut progress: ResMut<NarrativeProgress>,
) {
    let Some(ref world) = game_state.world else { return };
    let current_turn = world.meta.turn;

    // Only regenerate every 5 turns (history doesn't change rapidly)
    if current_turn < progress.last_history_turn + 5 && !narrative.dirty {
        return;
    }

    let ctx = build_world_context(&game_state);

    // Regenerate province histories
    for prov in &world.provinces {
        let text = history::province_history(prov.id, &prov.name, &game_state.events, &ctx);
        narrative.province_histories.insert(prov.id, text);
    }

    // Regenerate faction histories
    for faction in &world.factions {
        let text = history::faction_history(faction.id, &faction.name, &game_state.events, &ctx);
        narrative.faction_histories.insert(faction.id, text);
    }

    progress.last_history_turn = current_turn;
    narrative.dirty = false;
}

/// System: Generate turn summary sentences for new turns.
///
/// Produces a concise 1-3 sentence overview per turn, displayed as a header
/// in the event feed above that turn's events.
pub fn update_turn_summaries(
    game_state: Res<ClientGameState>,
    mut narrative: ResMut<NarrativeState>,
    mut progress: ResMut<NarrativeProgress>,
) {
    let Some(ref world) = game_state.world else { return };
    let current_turn = world.meta.turn;

    if current_turn <= progress.last_turn_summary {
        return;
    }

    let ctx = build_world_context(&game_state);

    // Generate summaries for any turns we missed (catch-up after reconnect)
    for turn in (progress.last_turn_summary + 1)..=current_turn {
        // Collect events for this specific turn
        let turn_events: Vec<_> = game_state.events.iter()
            .filter(|e| event_turn(e) == Some(turn))
            .cloned()
            .collect();

        let summary = history::turn_summary(turn, &turn_events, &ctx);
        narrative.turn_summaries.push((turn, summary));
    }

    // Cap at 200 turn summaries
    if narrative.turn_summaries.len() > 200 {
        let excess = narrative.turn_summaries.len() - 200;
        narrative.turn_summaries.drain(..excess);
    }

    progress.last_turn_summary = current_turn;
}

/// System: Generate war summary prose for all active wars.
///
/// Updated every 3 turns or when narrative is dirty.
pub fn update_war_summaries(
    game_state: Res<ClientGameState>,
    mut narrative: ResMut<NarrativeState>,
    mut progress: ResMut<NarrativeProgress>,
) {
    let Some(ref world) = game_state.world else { return };
    let current_turn = world.meta.turn;

    if current_turn < progress.last_war_summary_turn + 3 && !narrative.dirty {
        return;
    }

    let ctx = build_world_context(&game_state);

    // Find all active wars from realm data
    narrative.war_summaries.clear();
    for realm in &world.realms {
        for &enemy in &realm.at_war_with {
            let key = if realm.faction <= enemy {
                (realm.faction, enemy)
            } else {
                (enemy, realm.faction)
            };
            // Only generate once per war pair
            if !narrative.war_summaries.contains_key(&key) {
                let summary = history::war_summary(key.0, key.1, &game_state.events, &ctx);
                narrative.war_summaries.insert(key, summary);
            }
        }
    }

    progress.last_war_summary_turn = current_turn;
}

/// System: Update realm prosperity narrative for all factions.
///
/// Regenerated alongside histories (every 5 turns).
pub fn update_realm_prosperity(
    game_state: Res<ClientGameState>,
    mut narrative: ResMut<NarrativeState>,
    progress: Res<NarrativeProgress>,
) {
    let Some(ref world) = game_state.world else { return };
    let current_turn = world.meta.turn;

    // Piggyback on history regeneration schedule
    if current_turn < progress.last_history_turn + 5 && !narrative.dirty {
        return;
    }

    let ctx = build_world_context(&game_state);

    for faction in &world.factions {
        let controlled: Vec<u16> = world.provinces.iter()
            .filter(|p| p.controller == faction.id)
            .map(|p| p.id)
            .collect();
        let text = history::realm_prosperity(
            faction.id, &controlled, &game_state.events, &ctx,
        );
        narrative.realm_prosperity.insert(faction.id, text);
    }
}

/// System: Update intrigue narratives when new plot events arrive.
pub fn update_intrigue_narratives(
    game_state: Res<ClientGameState>,
    mut narrative: ResMut<NarrativeState>,
    progress: Res<NarrativeProgress>,
) {
    let Some(ref world) = game_state.world else { return };
    let current_turn = world.meta.turn;

    // Only regenerate every 3 turns
    if current_turn < progress.last_war_summary_turn + 3 && !narrative.dirty {
        return;
    }

    let ctx = build_world_context(&game_state);
    narrative.intrigue_narratives = history::intrigue_narrative(&game_state.events, &ctx);

    // Cap at 100 entries
    if narrative.intrigue_narratives.len() > 100 {
        let excess = narrative.intrigue_narratives.len() - 100;
        narrative.intrigue_narratives.drain(..excess);
    }
}

/// System: Update era summary overview every 10 turns.
///
/// Provides a high-level overview shown when nothing is selected.
pub fn update_era_summary(
    game_state: Res<ClientGameState>,
    mut narrative: ResMut<NarrativeState>,
    mut progress: ResMut<NarrativeProgress>,
) {
    let Some(ref world) = game_state.world else { return };
    let current_turn = world.meta.turn;

    // Only regenerate every 10 turns
    if current_turn < progress.last_era_summary_turn + 10 && !narrative.dirty {
        return;
    }

    let ctx = build_world_context(&game_state);

    let factions_alive = world.factions.len() as u32;
    let faction_provinces: Vec<(u8, u32)> = world.factions.iter()
        .map(|f| {
            let count = world.provinces.iter().filter(|p| p.controller == f.id).count() as u32;
            (f.id, count)
        })
        .collect();

    // Collect active wars from realm data
    let mut active_wars: Vec<(u8, u8)> = Vec::new();
    for realm in &world.realms {
        for &enemy in &realm.at_war_with {
            let key = if realm.faction <= enemy { (realm.faction, enemy) } else { (enemy, realm.faction) };
            if !active_wars.contains(&key) {
                active_wars.push(key);
            }
        }
    }

    narrative.era_summary_text = history::era_summary(
        current_turn, factions_alive, &active_wars, &faction_provinces,
        &game_state.events, &ctx,
    );

    progress.last_era_summary_turn = current_turn;
}

/// System: Update province religion narratives.
///
/// Piggybacks on history regeneration schedule (every 5 turns).
pub fn update_province_religions(
    game_state: Res<ClientGameState>,
    mut narrative: ResMut<NarrativeState>,
    progress: Res<NarrativeProgress>,
) {
    let Some(ref world) = game_state.world else { return };
    let current_turn = world.meta.turn;

    // Piggyback on history schedule
    if current_turn < progress.last_history_turn + 5 && !narrative.dirty {
        return;
    }

    let ctx = build_world_context(&game_state);

    for prov in &world.provinces {
        let religion = format!("{:?}", prov.religion);
        let text = history::religion_narrative(
            prov.id, &prov.name, &religion, &game_state.events, &ctx,
        );
        narrative.province_religion.insert(prov.id, text);
    }
}

/// System: Update diplomacy narratives between faction pairs.
///
/// Refreshes every 3 turns alongside war summaries.
pub fn update_diplomacy_narratives(
    game_state: Res<ClientGameState>,
    mut narrative: ResMut<NarrativeState>,
    progress: Res<NarrativeProgress>,
) {
    let Some(ref world) = game_state.world else { return };
    let current_turn = world.meta.turn;

    // Refresh every 3 turns
    if current_turn < progress.last_war_summary_turn + 3 && !narrative.dirty {
        return;
    }

    let ctx = build_world_context(&game_state);

    // Build war set for quick lookup
    let mut war_pairs: Vec<(u8, u8)> = Vec::new();
    for realm in &world.realms {
        for &enemy in &realm.at_war_with {
            let key = if realm.faction <= enemy { (realm.faction, enemy) } else { (enemy, realm.faction) };
            if !war_pairs.contains(&key) {
                war_pairs.push(key);
            }
        }
    }

    // Generate diplomacy narrative for every faction pair
    narrative.diplomacy_narratives.clear();
    let factions: Vec<u8> = world.factions.iter().map(|f| f.id).collect();
    for i in 0..factions.len() {
        for j in (i + 1)..factions.len() {
            let a = factions[i];
            let b = factions[j];
            let key = if a <= b { (a, b) } else { (b, a) };
            let at_war = war_pairs.contains(&key);
            let text = history::diplomacy_narrative(a, b, at_war, &game_state.events, &ctx);
            narrative.diplomacy_narratives.insert(key, text);
        }
    }
}

/// Extract the turn number from a GameEvent.
fn event_turn(event: &crown_ash_types::GameEvent) -> Option<u32> {
    use crown_ash_types::GameEvent::*;
    match event {
        Battle(r) => Some(r.turn),
        Harvest { turn, .. }
        | Famine { turn, .. }
        | PlagueOutbreak { turn, .. }
        | WarDeclared { turn, .. }
        | TreatySigned { turn, .. }
        | CharacterBorn { turn, .. }
        | CharacterDied { turn, .. }
        | ProvinceConquered { turn, .. }
        | SuccessionCrisis { turn, .. }
        | FactionEliminated { turn, .. }
        | Rebellion { turn, .. }
        | RealmSplit { turn, .. }
        | PlayerJoined { turn, .. }
        | ConstructionComplete { turn, .. }
        | PlotLaunched { turn, .. }
        | PlotSucceeded { turn, .. }
        | PlotDiscovered { turn, .. }
        | PlotFoiled { turn, .. }
        | TradeRouteEstablished { turn, .. }
        | TradeRouteDisrupted { turn, .. }
        | CharacterTombstoned { turn, .. }
        | ArmyAutoDisbanded { turn, .. }
        | ReligiousConversion { turn, .. }
        | Heresy { turn, .. }
        | Miracle { turn, .. }
        | SiegeStarted { turn, .. }
        | SiegeCompleted { turn, .. }
        | Friendship { turn, .. }
        | Rivalry { turn, .. }
        | MarriageAlliance { turn, .. } => Some(*turn),
    }
}
