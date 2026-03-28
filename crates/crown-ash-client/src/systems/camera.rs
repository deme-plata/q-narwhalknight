use bevy::prelude::*;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::input::ButtonInput;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Camera height above the XZ plane (fixed; panning moves X and Z only).
const CAMERA_Y: f32 = 30.0;

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

// ---------------------------------------------------------------------------
// Startup system — spawns the orthographic camera looking straight down.
// ---------------------------------------------------------------------------

pub fn setup_camera(mut commands: Commands) {
    commands.spawn((
        MapCamera,
        Camera3d::default(),
        Projection::from(OrthographicProjection {
            scale: 20.0,
            near: 0.1,
            far: 100.0,
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(3.0, CAMERA_Y, 0.0).looking_at(Vec3::new(3.0, 0.0, 0.0), Vec3::NEG_Z),
    ));
}

// ---------------------------------------------------------------------------
// Update system — WASD panning (moves camera on the XZ plane).
// ---------------------------------------------------------------------------

pub fn camera_pan(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut Transform, With<MapCamera>>,
) {
    let Ok(mut tf) = query.get_single_mut() else {
        return;
    };

    let dt = time.delta_secs();
    let mut delta = Vec3::ZERO;

    // The camera looks straight down (-Y). In screen space:
    //   "up"    on screen => camera -Z  (world north)
    //   "down"  on screen => camera +Z  (world south)
    //   "left"  on screen => camera -X
    //   "right" on screen => camera +X
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
        delta.z -= PAN_SPEED * dt;
    }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
        delta.z += PAN_SPEED * dt;
    }
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
        delta.x -= PAN_SPEED * dt;
    }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
        delta.x += PAN_SPEED * dt;
    }

    tf.translation += delta;

    // Keep Y fixed so the camera never tilts away from the map plane.
    tf.translation.y = CAMERA_Y;
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
