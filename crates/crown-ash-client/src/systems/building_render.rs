//! Building render system — loads GLB models and places them on province hexes.
//!
//! Maps each `Improvement` enum variant to one or more 3D building models.
//! Buildings are arranged in a clock-like ring within each hex tile.
//! Construction-in-progress buildings render semi-transparent.

use bevy::prelude::*;
use std::f32::consts::PI;

use crown_ash_types::Improvement;

use crate::components::building::{BuildingMarker, DecoMarker, UnderConstruction};
use crate::plugins::embedded_assets::EMBEDDED_PREFIX;
use crate::resources::game_state::ClientGameState;
use crate::systems::map_render::{PROVINCE_POSITIONS, province_elevation};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// How far from hex centre buildings are placed (fraction of hex radius 1.35).
const BUILDING_RING_RADIUS: f32 = 0.75;

/// Uniform scale for building models (they're ~1-2m in model space).
const BUILDING_SCALE: f32 = 0.18;

/// Y offset so buildings sit on top of the hex surface.
const BUILDING_Y: f32 = 0.01;

/// Maximum building slots per province (matches Improvement enum count).
const MAX_SLOTS: usize = 12;

// ---------------------------------------------------------------------------
// Asset catalog — maps Improvement → GLB path
// ---------------------------------------------------------------------------

/// Returns the embedded asset path for a given improvement type.
/// Paths use the `embedded://crown_ash_client/` prefix so assets are loaded
/// from the binary itself — no external `assets/` folder needed.
fn improvement_model_path(imp: Improvement) -> String {
    let rel = match imp {
        Improvement::Farmstead      => "models/village/barn.glb",
        Improvement::Mine           => "models/village/shed.glb",
        Improvement::Lumbercamp     => "models/village/woodpile.glb",
        Improvement::Quarry         => "models/village/longhouse.glb",
        Improvement::Stables        => "models/town/stable_block.glb",
        Improvement::Market         => "models/town/market_stall_a.glb",
        Improvement::Temple         => "models/religious/chapel.glb",
        Improvement::Fortification  => "models/town/town_gatehouse.glb",
        Improvement::University     => "models/town/merchant_house.glb",
        Improvement::Port           => "models/town/warehouse.glb",
        Improvement::Granary        => "models/village/granary.glb",
        Improvement::Hospital       => "models/town/workshop.glb",
    };
    format!("{EMBEDDED_PREFIX}{rel}")
}

// ---------------------------------------------------------------------------
// Building placement — clock positions within a hex
// ---------------------------------------------------------------------------

/// Returns local (x, z) offset for the Nth building slot within a hex.
/// Slots are arranged in a ring at `BUILDING_RING_RADIUS` from centre,
/// evenly spaced at 30° intervals (12 slots = full circle).
fn slot_offset(slot_index: usize) -> (f32, f32) {
    let angle = (slot_index as f32) * (2.0 * PI / MAX_SLOTS as f32) - PI / 2.0;
    let x = BUILDING_RING_RADIUS * angle.cos();
    let z = BUILDING_RING_RADIUS * angle.sin();
    (x, z)
}

/// Deterministic rotation for a building in a slot (face outward from centre).
fn slot_rotation(slot_index: usize) -> Quat {
    let angle = (slot_index as f32) * (2.0 * PI / MAX_SLOTS as f32) - PI / 2.0;
    Quat::from_rotation_y(-angle)
}

// ---------------------------------------------------------------------------
// Resource: preloaded scene handles
// ---------------------------------------------------------------------------

/// Holds preloaded GLB scene handles keyed by improvement type.
#[derive(Resource)]
pub struct BuildingAssets {
    scenes: Vec<(Improvement, Handle<Scene>)>,
}

impl BuildingAssets {
    fn get(&self, imp: Improvement) -> Option<Handle<Scene>> {
        self.scenes.iter()
            .find(|(i, _)| *i == imp)
            .map(|(_, h)| h.clone())
    }
}

// ---------------------------------------------------------------------------
// Resource: tracks which buildings are currently spawned per province
// ---------------------------------------------------------------------------

/// Tracks spawned building state to avoid respawning every frame.
#[derive(Resource, Default)]
pub struct BuildingState {
    /// For each province: the set of improvements currently rendered.
    province_buildings: Vec<Vec<Improvement>>,
    /// For each province: improvements under construction.
    province_construction: Vec<Vec<(Improvement, u32)>>,
}

// ---------------------------------------------------------------------------
// Startup system — load all GLB scenes
// ---------------------------------------------------------------------------

/// Loads all building model scenes at startup.
pub fn setup_building_assets(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    let improvements = [
        Improvement::Farmstead,
        Improvement::Mine,
        Improvement::Lumbercamp,
        Improvement::Quarry,
        Improvement::Stables,
        Improvement::Market,
        Improvement::Temple,
        Improvement::Fortification,
        Improvement::University,
        Improvement::Port,
        Improvement::Granary,
        Improvement::Hospital,
    ];

    let scenes: Vec<(Improvement, Handle<Scene>)> = improvements.iter()
        .map(|&imp| {
            let path = format!("{}#Scene0", improvement_model_path(imp));
            let handle: Handle<Scene> = asset_server.load(&path);
            (imp, handle)
        })
        .collect();

    commands.insert_resource(BuildingAssets { scenes });
    commands.insert_resource(BuildingState::default());
}

// ---------------------------------------------------------------------------
// Update system — spawn / despawn buildings based on province state
// ---------------------------------------------------------------------------

/// Syncs 3D building entities with each province's improvements list.
///
/// Runs every frame but only does work when the world state changes.
/// Compares current province improvements against the cached `BuildingState`.
pub fn update_buildings(
    mut commands: Commands,
    game_state: Res<ClientGameState>,
    assets: Res<BuildingAssets>,
    mut state: ResMut<BuildingState>,
    existing: Query<(Entity, &BuildingMarker)>,
) {
    let Some(ref world) = game_state.world else { return };

    // Ensure state vectors are the right size.
    if state.province_buildings.len() != world.provinces.len() {
        state.province_buildings.resize(world.provinces.len(), Vec::new());
        state.province_construction.resize(world.provinces.len(), Vec::new());
    }

    for prov in &world.provinces {
        let pid = prov.id as usize;
        if pid >= PROVINCE_POSITIONS.len() { continue; }

        let cached = &state.province_buildings[pid];
        let cached_construction = &state.province_construction[pid];

        // Check if anything changed.
        let improvements_match = cached.len() == prov.improvements.len()
            && cached.iter().zip(&prov.improvements).all(|(a, b)| a == b);
        let construction_match = cached_construction.len() == prov.construction_queue.len()
            && cached_construction.iter().zip(&prov.construction_queue).all(|(a, b)| a == b);

        if improvements_match && construction_match {
            continue; // No change for this province.
        }

        // Remove old building entities for this province.
        for (entity, marker) in existing.iter() {
            if marker.province_id == prov.id {
                commands.entity(entity).despawn_recursive();
            }
        }

        let (px, pz) = PROVINCE_POSITIONS[pid];
        let base_y = province_elevation(pid) + BUILDING_Y;
        let mut slot = 0usize;

        // Spawn completed improvements.
        for &imp in &prov.improvements {
            if slot >= MAX_SLOTS { break; }
            if let Some(scene) = assets.get(imp) {
                let (ox, oz) = slot_offset(slot);
                let rotation = slot_rotation(slot);

                commands.spawn((
                    BuildingMarker { province_id: prov.id, improvement: imp },
                    SceneRoot(scene),
                    Transform {
                        translation: Vec3::new(px + ox, base_y, pz + oz),
                        rotation,
                        scale: Vec3::splat(BUILDING_SCALE),
                    },
                ));
                slot += 1;
            }
        }

        // Spawn construction-in-progress buildings (semi-transparent later via material override).
        for &(imp, turns_left) in &prov.construction_queue {
            if slot >= MAX_SLOTS { break; }
            if let Some(scene) = assets.get(imp) {
                let (ox, oz) = slot_offset(slot);
                let rotation = slot_rotation(slot);

                // Spawn at smaller scale to indicate "under construction".
                let construction_scale = BUILDING_SCALE * 0.6;

                commands.spawn((
                    BuildingMarker { province_id: prov.id, improvement: imp },
                    UnderConstruction { turns_remaining: turns_left },
                    SceneRoot(scene),
                    Transform {
                        translation: Vec3::new(px + ox, base_y, pz + oz),
                        rotation,
                        scale: Vec3::splat(construction_scale),
                    },
                ));
                slot += 1;
            }
        }

        // Update cache.
        state.province_buildings[pid] = prov.improvements.clone();
        state.province_construction[pid] = prov.construction_queue.clone();
    }
}

// ---------------------------------------------------------------------------
// Decorative population buildings
// ---------------------------------------------------------------------------

/// Inner ring radius for decorative buildings (closer to centre than improvements).
const DECO_RING_RADIUS: f32 = 0.40;

/// Scale for decorative buildings (smaller than improvement buildings).
const DECO_SCALE: f32 = 0.12;

/// Maximum decorative building slots (inner ring, 6 positions at 60° intervals).
const MAX_DECO_SLOTS: usize = 6;

/// Village-tier decorative model sub-paths (population < 2000).
const VILLAGE_DECO: &[&str] = &[
    "models/village/peasant_house_a.glb",
    "models/village/peasant_house_b.glb",
    "models/village/peasant_house_c.glb",
    "models/village/well.glb",
    "models/village/chicken_coop.glb",
    "models/village/fenced_yard.glb",
];

/// Town-tier decorative model sub-paths (population >= 2000).
const TOWN_DECO: &[&str] = &[
    "models/town/merchant_house.glb",
    "models/town/tavern.glb",
    "models/town/storehouse.glb",
    "models/village/bakehouse.glb",
    "models/town/market_stall_b.glb",
    "models/town/market_stall_c.glb",
];

/// Holds preloaded scene handles for decorative buildings.
#[derive(Resource)]
pub struct DecoAssets {
    village: Vec<Handle<Scene>>,
    town: Vec<Handle<Scene>>,
}

/// Tracks spawned decorative state per province.
#[derive(Resource, Default)]
pub struct DecoState {
    /// Cached population tier per province (0=none, 1-6=count of decos).
    province_deco_count: Vec<u8>,
}

/// Loads decorative building scenes at startup.
pub fn setup_deco_assets(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    let village: Vec<Handle<Scene>> = VILLAGE_DECO.iter()
        .map(|p| asset_server.load(format!("{EMBEDDED_PREFIX}{p}#Scene0")))
        .collect();
    let town: Vec<Handle<Scene>> = TOWN_DECO.iter()
        .map(|p| asset_server.load(format!("{EMBEDDED_PREFIX}{p}#Scene0")))
        .collect();
    commands.insert_resource(DecoAssets { village, town });
    commands.insert_resource(DecoState::default());
}

/// Returns how many decorative buildings a province should have based on population.
fn deco_count_for_population(pop: u32) -> u8 {
    match pop {
        0..=499     => 0,
        500..=999   => 1,
        1000..=1999 => 2,
        2000..=3999 => 3,
        4000..=6999 => 4,
        7000..=9999 => 5,
        _           => 6,
    }
}

/// Returns (x, z) offset for decorative building slot within the inner ring.
fn deco_slot_offset(slot_index: usize) -> (f32, f32) {
    let angle = (slot_index as f32) * (2.0 * PI / MAX_DECO_SLOTS as f32);
    let x = DECO_RING_RADIUS * angle.cos();
    let z = DECO_RING_RADIUS * angle.sin();
    (x, z)
}

/// Deterministic rotation for a deco building (face centre).
fn deco_slot_rotation(slot_index: usize) -> Quat {
    let angle = (slot_index as f32) * (2.0 * PI / MAX_DECO_SLOTS as f32);
    Quat::from_rotation_y(PI - angle)
}

/// Spawns / despawns decorative buildings based on province population.
pub fn update_decorations(
    mut commands: Commands,
    game_state: Res<ClientGameState>,
    assets: Res<DecoAssets>,
    mut state: ResMut<DecoState>,
    existing: Query<(Entity, &DecoMarker)>,
) {
    let Some(ref world) = game_state.world else { return };

    if state.province_deco_count.len() != world.provinces.len() {
        state.province_deco_count.resize(world.provinces.len(), 0);
    }

    for prov in &world.provinces {
        let pid = prov.id as usize;
        if pid >= PROVINCE_POSITIONS.len() { continue; }

        let target_count = deco_count_for_population(prov.population);

        if state.province_deco_count[pid] == target_count {
            continue; // No change.
        }

        // Despawn existing decorations for this province.
        for (entity, marker) in existing.iter() {
            if marker.province_id == prov.id {
                commands.entity(entity).despawn_recursive();
            }
        }

        let (px, pz) = PROVINCE_POSITIONS[pid];
        let base_y = province_elevation(pid) + BUILDING_Y;

        // Pick models based on population tier.
        let is_town = prov.population >= 2000;

        for slot in 0..(target_count as usize).min(MAX_DECO_SLOTS) {
            // Deterministic model selection: hash province_id + slot.
            let model_idx = ((pid * 7 + slot * 13) % 6) as usize;

            let scene = if is_town && slot >= 2 {
                // Mix town buildings in for larger settlements.
                let town_idx = model_idx % assets.town.len();
                assets.town[town_idx].clone()
            } else {
                let village_idx = model_idx % assets.village.len();
                assets.village[village_idx].clone()
            };

            let (ox, oz) = deco_slot_offset(slot);
            let rotation = deco_slot_rotation(slot);

            commands.spawn((
                DecoMarker { province_id: prov.id },
                SceneRoot(scene),
                Transform {
                    translation: Vec3::new(px + ox, base_y, pz + oz),
                    rotation,
                    scale: Vec3::splat(DECO_SCALE),
                },
            ));
        }

        state.province_deco_count[pid] = target_count;
    }
}
