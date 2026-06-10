use bevy::prelude::*;

/// Marker component attached to every army icon entity.
/// Stores the army id for lookup into game state.
#[derive(Component)]
pub struct ArmyMarker {
    pub army_id: u32,
}
