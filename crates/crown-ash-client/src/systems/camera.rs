use bevy::prelude::*;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::input::ButtonInput;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Camera height above the focal point on the XZ plane.
const CAMERA_HEIGHT: f32 = 20.0;

/// Camera distance behind the focal point (creates tilt angle).
/// With height=20 and offset=12, the tilt is ~59° from horizontal.
const CAMERA_Z_OFFSET: f32 = 12.0;

/// Units-per-second when panning with WASD.
const PAN_SPEED: f32 = 20.0;

/// How much the orthographic scale changes per scroll tick.
const ZOOM_SPEED: f32 = 1.5;

/// Minimum orthographic projection scale (fully zoomed in).
const MIN_SCALE: f32 = 5.0;

/// Maximum orthographic projection scale (fully zoomed out).
const MAX_SCALE: f32 = 60.0;

/// Marker component so we can query for the main map camera specifically.
#[derive(Component)]
pub struct MapCamera;

/// Tracks the camera's focal point on the XZ plane.
#[derive(Resource)]
pub struct CameraFocus {
    pub x: f32,
    pub z: f32,
}

impl Default for CameraFocus {
    fn default() -> Self {
        // Centre the camera on the middle of the map.
        Self { x: 3.0, z: -1.5 }
    }
}

// ---------------------------------------------------------------------------
// Startup system — spawns the tilted orthographic camera.
// ---------------------------------------------------------------------------

pub fn setup_camera(mut commands: Commands) {
    let focus = CameraFocus::default();

    // Position camera above and behind the focal point for a tilted view.
    let cam_pos = Vec3::new(focus.x, CAMERA_HEIGHT, focus.z + CAMERA_Z_OFFSET);
    let look_at = Vec3::new(focus.x, 0.0, focus.z);

    commands.spawn((
        MapCamera,
        Camera3d::default(),
        Projection::from(OrthographicProjection {
            scale: 20.0,
            near: 0.1,
            far: 100.0,
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_translation(cam_pos).looking_at(look_at, Vec3::Y),
    ));

    commands.insert_resource(focus);

    // Lighting and ClearColor are set in setup_map (runs after setup_camera).
    // Sky-blue background — light enough to clearly distinguish from hex tiles.
    commands.insert_resource(ClearColor(Color::srgb(0.45, 0.55, 0.70)));
}

// ---------------------------------------------------------------------------
// Update system — WASD panning (moves camera focal point on XZ plane).
// ---------------------------------------------------------------------------

pub fn camera_pan(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut focus: ResMut<CameraFocus>,
    mut query: Query<&mut Transform, With<MapCamera>>,
) {
    let Ok(mut tf) = query.get_single_mut() else {
        return;
    };

    let dt = time.delta_secs();
    let mut dx = 0.0_f32;
    let mut dz = 0.0_f32;

    // Map screen directions to world XZ movement.
    // With the tilted camera, "up" on screen is roughly world -Z.
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
        dz -= PAN_SPEED * dt;
    }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
        dz += PAN_SPEED * dt;
    }
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
        dx -= PAN_SPEED * dt;
    }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
        dx += PAN_SPEED * dt;
    }

    if dx == 0.0 && dz == 0.0 {
        return;
    }

    // Update focal point.
    focus.x += dx;
    focus.z += dz;

    // Position camera above and behind the focal point.
    tf.translation = Vec3::new(focus.x, CAMERA_HEIGHT, focus.z + CAMERA_Z_OFFSET);
    let look_at = Vec3::new(focus.x, 0.0, focus.z);
    *tf = Transform::from_translation(tf.translation).looking_at(look_at, Vec3::Y);
}

// ---------------------------------------------------------------------------
// Update system — scroll-wheel zoom (adjusts orthographic scale).
// ---------------------------------------------------------------------------

pub fn camera_zoom(
    mut scroll_events: EventReader<MouseWheel>,
    mut query: Query<&mut Projection, With<MapCamera>>,
) {
    let Ok(mut projection) = query.get_single_mut() else {
        return;
    };

    let Projection::Orthographic(ref mut ortho) = *projection else {
        return;
    };

    for ev in scroll_events.read() {
        let scroll_amount = match ev.unit {
            MouseScrollUnit::Line => ev.y,
            MouseScrollUnit::Pixel => ev.y / 120.0,
        };

        // Scroll up => zoom in (smaller scale), scroll down => zoom out.
        ortho.scale -= scroll_amount * ZOOM_SPEED;
        ortho.scale = ortho.scale.clamp(MIN_SCALE, MAX_SCALE);
    }
}
