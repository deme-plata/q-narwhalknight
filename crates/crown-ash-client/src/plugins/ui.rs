//! Crown & Ash UI plugin — wires egui panels and action submission into Bevy.
//!
//! Adds the `EguiPlugin` and registers all UI systems:
//! - `top_bar` — turn counter, faction count, population, connection indicator
//! - `faction_list` — left panel with all factions overview and click-to-select
//! - `detail_panel` — province / faction / character / army details
//! - `event_feed` — scrolling narrative event log
//! - `minimap` — small map overview with clickable provinces
//! - `action_buttons` — player action submission (all 17 game actions)
//! - `join_dialog` — "Join Game" overlay for new players
//! - `keyboard_shortcuts` — ESC/Tab/1-7 for quick navigation

use bevy::prelude::*;
use bevy_egui::EguiPlugin;

use crate::resources::narrative_state::{DialogState, NarrativeState, ToastState};
use crate::systems::{action_submit, narrative_update, ui_panels};

/// Plugin that sets up the entire Crown & Ash egui-based UI layer.
pub struct CrownAshUiPlugin;

impl Plugin for CrownAshUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin)
            .init_resource::<action_submit::ActionState>()
            .init_resource::<ui_panels::JoinState>()
            .init_resource::<NarrativeState>()
            .init_resource::<DialogState>()
            .init_resource::<ToastState>()
            .init_resource::<narrative_update::NarrativeProgress>()
            // Narrative update systems (run before UI so text is ready)
            .add_systems(Update, narrative_update::update_event_narratives)
            .add_systems(Update, narrative_update::update_chronicles)
            .add_systems(Update, narrative_update::update_histories)
            .add_systems(Update, narrative_update::update_turn_summaries)
            .add_systems(Update, narrative_update::update_war_summaries)
            .add_systems(Update, narrative_update::update_realm_prosperity)
            .add_systems(Update, narrative_update::update_intrigue_narratives)
            .add_systems(Update, narrative_update::update_era_summary)
            .add_systems(Update, narrative_update::update_province_religions)
            .add_systems(Update, narrative_update::update_diplomacy_narratives)
            // UI panels
            .add_systems(Update, ui_panels::top_bar)
            .add_systems(Update, ui_panels::faction_list)
            .add_systems(Update, ui_panels::detail_panel)
            .add_systems(Update, ui_panels::event_feed)
            .add_systems(Update, ui_panels::minimap)
            .add_systems(Update, ui_panels::join_dialog)
            .add_systems(Update, ui_panels::keyboard_shortcuts)
            .add_systems(Update, ui_panels::dialog_bubbles)
            .add_systems(Update, ui_panels::notification_toasts)
            .add_systems(Update, action_submit::action_buttons)
            .add_systems(Update, ui_panels::hover_tooltip);
    }
}
