//! Building marker component for 3D building models placed on province hexes.

use bevy::prelude::*;
use crown_ash_types::Improvement;

/// Tags a spawned 3D building scene with its province and improvement type.
#[derive(Component)]
pub struct BuildingMarker {
    pub province_id: u16,
    pub improvement: Improvement,
}

/// Tags buildings that are under construction (shown semi-transparent / scaffolded).
#[derive(Component)]
pub struct UnderConstruction {
    pub turns_remaining: u32,
}

/// Tags a decorative building spawned based on population (not an Improvement).
#[derive(Component)]
pub struct DecoMarker {
    pub province_id: u16,
}
