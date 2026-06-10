use bevy::prelude::*;

/// Marker component for province name labels floating above the map.
#[derive(Component)]
pub struct ProvinceLabel {
    pub province_id: u16,
}

/// Marker for the selection highlight ring entity.
#[derive(Component)]
pub struct SelectionRing;

/// Marker for army movement path lines drawn on the map.
/// Each entity represents the full path line for one army.
#[derive(Component)]
pub struct MovementPath {
    pub army_id: u32,
}
