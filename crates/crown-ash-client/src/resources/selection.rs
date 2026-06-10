//! User selection state — tracks what the player has clicked / hovered.

use bevy::prelude::*;

/// Tracks the currently selected game entities.
///
/// UI and map systems read this to highlight the selected province, show the
/// faction panel, display character details, etc.  Input systems are the
/// primary writers.
#[derive(Resource, Default)]
pub struct Selection {
    /// Currently selected province (by `ProvinceId`).
    pub province: Option<u16>,
    /// Currently selected faction (by `FactionId`).
    pub faction: Option<u8>,
    /// Currently selected character (by `CharacterId`).
    pub character: Option<u32>,
    /// Currently selected army (by `ArmyId`).
    pub army: Option<u32>,
    /// Province currently under the mouse cursor (for hover tooltips).
    pub hovered_province: Option<u16>,
    /// Screen-space position of the cursor (for tooltip placement).
    pub cursor_screen_pos: Option<(f32, f32)>,
}
