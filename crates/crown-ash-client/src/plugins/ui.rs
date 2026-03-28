//! Crown & Ash UI plugin — wires egui panels and action submission into Bevy.
//!
//! Adds the `EguiPlugin` and registers all UI systems:
//! - `top_bar` — turn counter, faction count, population, connection indicator
//! - `detail_panel` — province / faction / character / army details
//! - `event_feed` — scrolling narrative event log
//! - `action_buttons` — player action submission (raise army, etc.)

use bevy::prelude::*;
use bevy_egui::EguiPlugin;

use crate::systems::{action_submit, ui_panels};

/// Plugin that sets up the entire Crown & Ash egui-based UI layer.
pub struct CrownAshUiPlugin;

impl Plugin for CrownAshUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin)
            .init_resource::<action_submit::ActionState>()
            .add_systems(
                Update,
                (
                    ui_panels::top_bar,
                    ui_panels::detail_panel,
                    ui_panels::event_feed,
                    action_submit::action_buttons,
                ),
            );
    }
}
