use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::input::ButtonInput;
use std::f32::consts::PI;

use crate::components::province::ProvinceMarker;
use crate::components::army::ArmyMarker;
use crate::resources::game_state::ClientGameState;
use crate::resources::selection::Selection;
use crate::systems::camera::MapCamera;

// ---------------------------------------------------------------------------
// Province layout data
// ---------------------------------------------------------------------------

/// Fixed province positions on the XZ plane (x, z).
/// Derived from the geographical cluster layout in crown-ash-sim/src/map.rs.
pub const PROVINCE_POSITIONS: [(f32, f32); 25] = [
    // Frost Marches (north) — provinces 0-3
    (-4.5, -9.0),  // 0: Frosthold (Mountains)
    (-1.5, -9.0),  // 1: Winterfell Vale (Hills)
    ( 1.5, -9.0),  // 2: Icemere (Marsh)
    ( 4.5, -9.0),  // 3: Stormwatch (Coastal)
    // Vale Princes (southeast) — provinces 4-6
    ( 0.0,  3.0),  // 4: Goldhaven (Plains)
    ( 3.0,  3.0),  // 5: Thornwall (Hills)
    ( 6.0,  3.0),  // 6: Ravensgate (Forest)
    // Ashen Crown (center) — provinces 7-10
    (-1.5, -3.0),  // 7: Ashenmere (Plains)
    ( 1.5, -3.0),  // 8: Crownspire (Hills)
    ( 1.5,  0.0),  // 9: Embervale (River)
    ( 4.5,  0.0),  // 10: Kingsreach (Plains)
    // Ember Church (southwest) — provinces 11-13
    (-4.5,  0.0),  // 11: Sanctum (Hills)
    (-1.5,  3.0),  // 12: Pyrelight (Plains)
    ( 0.0,  6.0),  // 13: Candlekeep (Forest)
    // Salt League (east) — provinces 14-17
    ( 7.5,  0.0),  // 14: Saltmere (Coastal)
    ( 9.0,  3.0),  // 15: Tidehollow (Coastal)
    (10.5,  0.0),  // 16: Coinport (Coastal)
    ( 9.0,  6.0),  // 17: Warehouse Row (Plains)
    // Black Abbey (west) — provinces 18-20
    (-7.5, -6.0),  // 18: Shadowmere (Forest)
    (-7.5, -3.0),  // 19: Whispering Cloister (Hills)
    (-4.5, -3.0),  // 20: Veilstone (Mountains)
    // Red Steppe (northeast) — provinces 21-24
    ( 7.5, -9.0),  // 21: Khanstead (Plains)
    ( 7.5, -6.0),  // 22: Windbreak (Desert)
    (10.5, -6.0),  // 23: Dustmane (Desert)
    (10.5, -3.0),  // 24: Redhorn (Plains)
];

/// Adjacency list — pairs of province IDs that share a border.
pub const ADJACENCY: &[(u16, u16)] = &[
    (0,1),(0,2),(0,18),   (1,2),(1,7),   (2,3),(2,8),   (3,8),(3,21),
    (4,5),(4,9),(4,13),   (5,6),(5,10),(5,14),   (6,10),(6,15),
    (7,8),(7,9),(7,11),   (8,9),(8,20),(8,21),   (9,10),
    (10,5),(10,6),(10,12),
    (11,12),(11,19),   (12,13),(12,10),   (13,4),(13,17),
    (14,15),(14,16),   (15,16),   (16,17),   (17,24),
    (18,19),(18,20),   (19,11),(19,20),   (20,8),
    (21,22),   (22,23),(22,24),   (23,24),
];

/// Starting faction for each province (used as fallback when no world data is present).
const DEFAULT_FACTION: [u8; 25] = [
    4, 4, 4, 4,    // 0-3:   Frost Marches
    1, 1, 1,        // 4-6:   Vale Princes
    0, 0, 0, 0,    // 7-10:  Ashen Crown
    2, 2, 2,        // 11-13: Ember Church
    3, 3, 3, 3,    // 14-17: Salt League
    6, 6, 6,        // 18-20: Black Abbey
    5, 5, 5, 5,    // 21-24: Red Steppe
];

/// Default faction colors (r,g,b). Index = faction id.
/// These are used when no world snapshot is available yet.
const DEFAULT_FACTION_COLORS: [[u8; 3]; 7] = [
    [200,  50,  50],  // 0: Ashen Crown — crimson
    [ 50,  50, 200],  // 1: Vale Princes — blue
    [200, 180,  30],  // 2: Ember Church — gold
    [ 40, 180, 180],  // 3: Salt League — teal
    [180, 180, 220],  // 4: Frost Marches — pale ice-blue
    [180,  80,  40],  // 5: Red Steppe — rust
    [ 90,  40, 130],  // 6: Black Abbey — dark purple
];

/// Terrain enum indices (matching crown_ash_types::Terrain order).
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
enum Terrain {
    Plains = 0,
    Hills,
    Mountains,
    Forest,
    Marsh,
    Desert,
    Coastal,
    River,
}

/// Default terrain per province.
const DEFAULT_TERRAIN: [Terrain; 25] = [
    Terrain::Mountains, Terrain::Hills, Terrain::Marsh, Terrain::Coastal,   // 0-3
    Terrain::Plains, Terrain::Hills, Terrain::Forest,                       // 4-6
    Terrain::Plains, Terrain::Hills, Terrain::River, Terrain::Plains,       // 7-10
    Terrain::Hills, Terrain::Plains, Terrain::Forest,                       // 11-13
    Terrain::Coastal, Terrain::Coastal, Terrain::Coastal, Terrain::Plains,  // 14-17
    Terrain::Forest, Terrain::Hills, Terrain::Mountains,                    // 18-20
    Terrain::Plains, Terrain::Desert, Terrain::Desert, Terrain::Plains,     // 21-24
];

// ---------------------------------------------------------------------------
// Hex radius
// ---------------------------------------------------------------------------

const HEX_RADIUS: f32 = 1.35;

// ---------------------------------------------------------------------------
// Mesh builders
// ---------------------------------------------------------------------------

/// Creates a flat hexagonal mesh on the XZ plane (Y = 0).
/// 7 vertices: centre + 6 outer points; 6 triangles.
fn build_hex_mesh(radius: f32) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(7);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(7);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(7);
    let mut indices: Vec<u32> = Vec::with_capacity(18);

    // Centre vertex
    positions.push([0.0, 0.0, 0.0]);
    normals.push([0.0, 1.0, 0.0]);
    uvs.push([0.5, 0.5]);

    // 6 outer vertices — flat-top hex (first vertex at +X).
    for i in 0..6 {
        let angle = (i as f32) * PI / 3.0;
        let x = radius * angle.cos();
        let z = radius * angle.sin();
        positions.push([x, 0.0, z]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([0.5 + 0.5 * angle.cos(), 0.5 + 0.5 * angle.sin()]);
    }

    // 6 triangles fan from centre.
    for i in 0u32..6 {
        let next = (i + 1) % 6;
        indices.push(0);
        indices.push(i + 1);
        indices.push(next + 1);
    }

    Mesh::new(PrimitiveTopology::TriangleList, bevy::render::render_asset::RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
}

/// Creates a line-list mesh for all adjacency edges on the XZ plane (Y slightly above 0 to avoid z-fight).
fn build_adjacency_lines_mesh() -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(ADJACENCY.len() * 2);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(ADJACENCY.len() * 2);

    let y = 0.02; // slightly above province hexes to stay visible
    for &(a, b) in ADJACENCY.iter() {
        let (ax, az) = PROVINCE_POSITIONS[a as usize];
        let (bx, bz) = PROVINCE_POSITIONS[b as usize];
        positions.push([ax, y, az]);
        positions.push([bx, y, bz]);
        normals.push([0.0, 1.0, 0.0]);
        normals.push([0.0, 1.0, 0.0]);
    }

    Mesh::new(PrimitiveTopology::LineList, bevy::render::render_asset::RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
}

// ---------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------

/// Terrain-based tint applied multiplicatively to the faction base colour.
fn terrain_tint(terrain: Terrain) -> Color {
    match terrain {
        Terrain::Plains    => Color::srgb(0.95, 1.00, 0.90),
        Terrain::Hills     => Color::srgb(0.85, 0.82, 0.70),
        Terrain::Mountains => Color::srgb(0.65, 0.65, 0.70),
        Terrain::Forest    => Color::srgb(0.55, 0.80, 0.50),
        Terrain::Marsh     => Color::srgb(0.60, 0.75, 0.65),
        Terrain::Desert    => Color::srgb(0.95, 0.88, 0.65),
        Terrain::Coastal   => Color::srgb(0.70, 0.85, 1.00),
        Terrain::River     => Color::srgb(0.65, 0.80, 0.95),
    }
}

/// Blend faction colour with terrain tint.
fn province_color(faction_rgb: [u8; 3], terrain: Terrain) -> Color {
    let tint = terrain_tint(terrain);
    let LinearRgba { red: tr, green: tg, blue: tb, .. } = tint.to_linear();

    let fr = faction_rgb[0] as f32 / 255.0;
    let fg = faction_rgb[1] as f32 / 255.0;
    let fb = faction_rgb[2] as f32 / 255.0;

    Color::srgb(
        (fr * tr).clamp(0.0, 1.0),
        (fg * tg).clamp(0.0, 1.0),
        (fb * tb).clamp(0.0, 1.0),
    )
}

// ---------------------------------------------------------------------------
// Startup system — spawn province hexes and adjacency lines
// ---------------------------------------------------------------------------

pub fn setup_map(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Shared hex mesh handle — every province uses the same shape.
    let hex_mesh = meshes.add(build_hex_mesh(HEX_RADIUS));

    // Spawn province hexes.
    for (i, &(px, pz)) in PROVINCE_POSITIONS.iter().enumerate() {
        let faction_id = DEFAULT_FACTION[i];
        let terrain = DEFAULT_TERRAIN[i];
        let color = province_color(DEFAULT_FACTION_COLORS[faction_id as usize], terrain);

        let material = materials.add(StandardMaterial {
            base_color: color,
            unlit: true, // flat-shaded strategy map; no need for lighting complexity
            double_sided: true,
            ..default()
        });

        commands.spawn((
            ProvinceMarker { province_id: i as u16 },
            Mesh3d(hex_mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_xyz(px, 0.0, pz),
        ));
    }

    // Spawn adjacency lines as a single mesh entity.
    let line_mesh = meshes.add(build_adjacency_lines_mesh());
    let line_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.3, 0.3, 0.3, 0.6),
        unlit: true,
        ..default()
    });

    commands.spawn((
        Mesh3d(line_mesh),
        MeshMaterial3d(line_mat),
        Transform::IDENTITY,
    ));

    // Simple ambient light so unlit materials render consistently.
    commands.spawn((
        DirectionalLight {
            illuminance: 5000.0,
            ..default()
        },
        Transform::from_xyz(5.0, 20.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

// ---------------------------------------------------------------------------
// Update system — re-colour provinces when world state changes
// ---------------------------------------------------------------------------

pub fn update_map_colors(
    game_state: Res<ClientGameState>,
    query: Query<(&ProvinceMarker, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Some(ref world) = game_state.world else {
        return;
    };

    for (marker, mat_handle) in query.iter() {
        let pid = marker.province_id as usize;
        if pid >= world.provinces.len() {
            continue;
        }

        let province = &world.provinces[pid];
        let controller = province.controller as usize;

        // Resolve faction colour.
        let faction_rgb = if controller < world.factions.len() {
            world.factions[controller].color_rgb
        } else if controller < DEFAULT_FACTION_COLORS.len() {
            DEFAULT_FACTION_COLORS[controller]
        } else {
            [128, 128, 128]
        };

        // Resolve terrain — use the index into our local enum.
        let terrain = if pid < DEFAULT_TERRAIN.len() {
            DEFAULT_TERRAIN[pid]
        } else {
            Terrain::Plains
        };

        let new_color = province_color(faction_rgb, terrain);

        if let Some(mat) = materials.get_mut(mat_handle) {
            mat.base_color = new_color;
        }
    }
}

// ---------------------------------------------------------------------------
// Update system — sync army entities with world state
// ---------------------------------------------------------------------------

/// Vertical offset so army icons float above the hex surface.
const ARMY_Y: f32 = 0.3;

/// Army icons are offset from province centre to avoid overlap with the hex.
const ARMY_OFFSET_X: f32 = 0.6;
const ARMY_OFFSET_Z: f32 = 0.6;

pub fn update_armies(
    mut commands: Commands,
    game_state: Res<ClientGameState>,
    existing: Query<(Entity, &ArmyMarker)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Some(ref world) = game_state.world else {
        return;
    };

    // Collect IDs of armies currently in the world.
    let live_ids: std::collections::HashSet<u32> = world.armies.iter().map(|a| a.id).collect();

    // Despawn entities whose army no longer exists.
    for (entity, marker) in existing.iter() {
        if !live_ids.contains(&marker.army_id) {
            commands.entity(entity).despawn();
        }
    }

    // Collect IDs of entities we already have.
    let existing_ids: std::collections::HashSet<u32> = existing.iter().map(|(_, m)| m.army_id).collect();

    // Spawn new armies.
    let cube_mesh = meshes.add(Cuboid::new(0.4, 0.4, 0.4));

    for army in world.armies.iter() {
        if existing_ids.contains(&army.id) {
            continue;
        }

        let loc = army.location as usize;
        if loc >= PROVINCE_POSITIONS.len() {
            continue;
        }

        let (px, pz) = PROVINCE_POSITIONS[loc];

        // Resolve faction colour for the army icon.
        let owner = army.owner_faction as usize;
        let rgb = if owner < world.factions.len() {
            world.factions[owner].color_rgb
        } else if owner < DEFAULT_FACTION_COLORS.len() {
            DEFAULT_FACTION_COLORS[owner]
        } else {
            [200, 200, 200]
        };

        let color = Color::srgb(rgb[0] as f32 / 255.0, rgb[1] as f32 / 255.0, rgb[2] as f32 / 255.0);

        let mat = materials.add(StandardMaterial {
            base_color: color,
            unlit: true,
            ..default()
        });

        commands.spawn((
            ArmyMarker { army_id: army.id },
            Mesh3d(cube_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(px + ARMY_OFFSET_X, ARMY_Y, pz + ARMY_OFFSET_Z),
        ));
    }
}

// ---------------------------------------------------------------------------
// Update system — left-click to select a province
// ---------------------------------------------------------------------------

pub fn handle_province_click(
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<MapCamera>>,
    mut selection: ResMut<Selection>,
) {
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }

    let Ok(window) = windows.get_single() else {
        return;
    };

    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };

    let Ok((camera, cam_transform)) = camera_q.get_single() else {
        return;
    };

    // Cast a ray from the camera through the cursor into the world.
    let Ok(ray) = camera.viewport_to_world(cam_transform, cursor_pos) else {
        return;
    };

    // Find intersection with Y=0 plane.
    // ray: origin + t * direction, solve for y=0 => t = -origin.y / direction.y
    if ray.direction.y.abs() < 1e-6 {
        return; // ray is parallel to the plane
    }

    let t = -ray.origin.y / ray.direction.y;
    if t < 0.0 {
        return; // intersection is behind the camera
    }

    let hit = ray.origin + t * *ray.direction;
    let hit_x = hit.x;
    let hit_z = hit.z;

    // Find the province closest to the hit point (within HEX_RADIUS).
    let mut best: Option<(u16, f32)> = None;
    for (i, &(px, pz)) in PROVINCE_POSITIONS.iter().enumerate() {
        let dx = hit_x - px;
        let dz = hit_z - pz;
        let dist_sq = dx * dx + dz * dz;
        if dist_sq < HEX_RADIUS * HEX_RADIUS {
            if best.map_or(true, |(_, bd)| dist_sq < bd) {
                best = Some((i as u16, dist_sq));
            }
        }
    }

    if let Some((pid, _)) = best {
        selection.province = Some(pid);
    }
}
