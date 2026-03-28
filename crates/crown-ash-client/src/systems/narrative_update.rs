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
use crate::resources::narrative_state::{EventNarrative, NarrativeState};

/// Tracks how many events we've already processed to avoid reprocessing.
#[derive(Resource)]
pub struct NarrativeProgress {
    /// Number of events already converted to narrative.
    pub events_processed: usize,
    /// Last turn for which chronicles were updated.
    pub last_chronicle_turn: u32,
    /// Last turn for which histories were regenerated.
    pub last_history_turn: u32,
}

impl Default for NarrativeProgress {
    fn default() -> Self {
        Self {
            events_processed: 0,
            last_chronicle_turn: 0,
            last_history_turn: 0,
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
        narrative.event_narratives.push(EventNarrative::from(&result));
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
