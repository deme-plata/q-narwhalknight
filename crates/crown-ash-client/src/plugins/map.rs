use bevy::prelude::*;
use crate::systems::{camera, map_render};
use crate::resources::selection::Selection;

/// Plugin that sets up the Crown & Ash strategic map: province hexes, adjacency
/// lines, army icons, camera controls, and province selection via click.
pub struct CrownAshMapPlugin;

impl Plugin for CrownAshMapPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Selection>()
            .add_systems(
                Startup,
                (camera::setup_camera, map_render::setup_map),
            )
            .add_systems(
                Update,
                (
                    camera::camera_pan,
                    camera::camera_zoom,
                    map_render::update_map_colors,
                    map_render::update_armies,
                    map_render::handle_province_click,
                ),
            );
    }
}
