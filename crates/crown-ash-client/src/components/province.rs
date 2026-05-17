use bevy::prelude::*;

/// Marker component attached to every province hex entity.
/// Stores the province index (0..24) for lookup into game state.
#[derive(Component)]
pub struct ProvinceMarker {
    pub province_id: u16,
}
