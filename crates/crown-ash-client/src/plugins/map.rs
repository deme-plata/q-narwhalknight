use bevy::prelude::*;
use crate::systems::{camera, map_render, building_render, terrain_features};
use crate::resources::selection::Selection;

/// Plugin that sets up the Crown & Ash strategic map: province hexes, adjacency
/// lines, army icons, name labels, selection ring, camera controls, click,
/// 3D building models, and procedural terrain features.
pub struct CrownAshMapPlugin;

impl Plugin for CrownAshMapPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Selection>()
            .add_systems(
                Startup,
                (
                    camera::setup_camera,
                    map_render::setup_map,
                    building_render::setup_building_assets,
                    building_render::setup_deco_assets,
                    terrain_features::setup_terrain_features,
                ).chain(), // chain ensures meshes are ready before spawning
            )
            .add_systems(
                Startup,
                terrain_features::spawn_terrain_features
                    .after(terrain_features::setup_terrain_features),
            )
            .add_systems(
                Update,
                (
                    camera::camera_pan,
                    camera::camera_zoom,
                    map_render::update_hover,
                    map_render::update_map_colors,
                    map_render::update_armies,
                    map_render::handle_province_click,
                    map_render::update_selection_ring,
                    map_render::update_province_labels,
                    map_render::update_movement_paths,
                    map_render::animate_army_movement,
                    building_render::update_buildings,
                    building_render::update_decorations,
                    terrain_features::animate_water,
                    map_render::update_siege_markers,
                    map_render::animate_siege_markers,
                ),
            );
    }
}
