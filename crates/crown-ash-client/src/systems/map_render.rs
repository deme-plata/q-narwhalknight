use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::input::ButtonInput;
use std::f32::consts::PI;

use crate::components::province::ProvinceMarker;
use crate::components::army::ArmyMarker;
use crate::components::label::{MovementPath, ProvinceLabel, SelectionRing};
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
// Terrain elevation
// ---------------------------------------------------------------------------

/// Y offset for each terrain type, giving the map 3D depth.
fn terrain_elevation(terrain: Terrain) -> f32 {
    match terrain {
        Terrain::Mountains => 0.45,
        Terrain::Hills     => 0.20,
        Terrain::Forest    => 0.08,
        Terrain::Plains    => 0.0,
        Terrain::Desert    => 0.0,
        Terrain::River     => -0.05,
        Terrain::Marsh     => -0.08,
        Terrain::Coastal   => -0.05,
    }
}

/// Returns the Y height for a province by index (using default terrain).
pub fn province_elevation(province_id: usize) -> f32 {
    if province_id < DEFAULT_TERRAIN.len() {
        terrain_elevation(DEFAULT_TERRAIN[province_id])
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Mesh builders
// ---------------------------------------------------------------------------

/// Centre dome height for each terrain type (how much the centre vertex
/// is raised above the edges, creating a gentle dome shape).
fn terrain_dome_height(terrain: Terrain) -> f32 {
    match terrain {
        Terrain::Mountains => 0.35,
        Terrain::Hills     => 0.15,
        Terrain::Forest    => 0.05,
        _ => 0.0, // flat for plains, desert, coastal, river, marsh
    }
}

/// Creates a hexagonal mesh on the XZ plane with an optional raised centre
/// (dome effect for hills/mountains). 7 vertices: centre + 6 outer points.
fn build_hex_mesh(radius: f32, dome: f32) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(7);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(7);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(7);
    let mut indices: Vec<u32> = Vec::with_capacity(18);

    // Centre vertex — raised by dome height.
    positions.push([0.0, dome, 0.0]);
    normals.push([0.0, 1.0, 0.0]);
    uvs.push([0.5, 0.5]);

    // 6 outer vertices — flat-top hex (first vertex at +X), at Y=0.
    for i in 0..6 {
        let angle = (i as f32) * PI / 3.0;
        let x = radius * angle.cos();
        let z = radius * angle.sin();
        positions.push([x, 0.0, z]);
        // Normals tilt outward slightly when domed.
        if dome > 0.001 {
            let nx = -x * dome / radius;
            let nz = -z * dome / radius;
            let len = (nx * nx + 1.0 + nz * nz).sqrt();
            normals.push([nx / len, 1.0 / len, nz / len]);
        } else {
            normals.push([0.0, 1.0, 0.0]);
        }
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

    for &(a, b) in ADJACENCY.iter() {
        let (ax, az) = PROVINCE_POSITIONS[a as usize];
        let (bx, bz) = PROVINCE_POSITIONS[b as usize];
        let ya = province_elevation(a as usize) + 0.02;
        let yb = province_elevation(b as usize) + 0.02;
        positions.push([ax, ya, az]);
        positions.push([bx, yb, bz]);
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
    // Pre-build hex meshes for each terrain type (different dome heights).
    use std::collections::HashMap;
    let mut terrain_meshes: HashMap<u8, Handle<Mesh>> = HashMap::new();
    for terrain_val in [
        Terrain::Plains, Terrain::Hills, Terrain::Mountains,
        Terrain::Forest, Terrain::Marsh, Terrain::Desert,
        Terrain::Coastal, Terrain::River,
    ] {
        let dome = terrain_dome_height(terrain_val);
        let mesh = meshes.add(build_hex_mesh(HEX_RADIUS, dome));
        terrain_meshes.insert(terrain_val as u8, mesh);
    }

    // Spawn province hexes with terrain-appropriate mesh and elevation.
    for (i, &(px, pz)) in PROVINCE_POSITIONS.iter().enumerate() {
        let faction_id = DEFAULT_FACTION[i];
        let terrain = DEFAULT_TERRAIN[i];
        let color = province_color(DEFAULT_FACTION_COLORS[faction_id as usize], terrain);
        let elevation = terrain_elevation(terrain);

        let material = materials.add(StandardMaterial {
            base_color: color,
            perceptual_roughness: 0.85,
            double_sided: true,
            ..default()
        });

        let hex_mesh = terrain_meshes[&(terrain as u8)].clone();

        commands.spawn((
            ProvinceMarker { province_id: i as u16 },
            Mesh3d(hex_mesh),
            MeshMaterial3d(material),
            Transform::from_xyz(px, elevation, pz),
        ));
    }

    // Spawn adjacency lines as a single mesh entity.
    let line_mesh = meshes.add(build_adjacency_lines_mesh());
    let line_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.15, 0.12, 0.08, 0.8),
        unlit: true,
        ..default()
    });

    commands.spawn((
        Mesh3d(line_mesh),
        MeshMaterial3d(line_mat),
        Transform::IDENTITY,
    ));

    // Large ground plane beneath the map for visual grounding.
    let ground_mesh = meshes.add(Mesh::from(Plane3d::default().mesh().size(80.0, 80.0)));
    let ground_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.28, 0.32, 0.22), // muted olive green
        perceptual_roughness: 0.95,
        ..default()
    });
    commands.spawn((
        Mesh3d(ground_mesh),
        MeshMaterial3d(ground_mat),
        Transform::from_xyz(2.0, -0.5, -2.0), // centred beneath the map, slightly below hexes
    ));

    // Sun light — angled to create shadows on domed terrain and 3D buildings.
    commands.spawn((
        DirectionalLight {
            illuminance: 15_000.0,
            shadows_enabled: true,
            color: Color::srgb(1.0, 0.96, 0.88),
            ..default()
        },
        Transform::from_xyz(8.0, 25.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Ambient light — bright enough to clearly see all terrain.
    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.65, 0.70, 0.80),
        brightness: 600.0,
    });

    // Province name labels — text billboards floating above each hex.
    let province_names = [
        "Frosthold", "Winterfell Vale", "Icemere", "Stormwatch",
        "Goldhaven", "Thornwall", "Ravensgate",
        "Ashenmere", "Crownspire", "Embervale", "Kingsreach",
        "Sanctum", "Pyrelight", "Candlekeep",
        "Saltmere", "Tidehollow", "Coinport", "Warehouse Row",
        "Shadowmere", "Whispering Cloister", "Veilstone",
        "Khanstead", "Windbreak", "Dustmane", "Redhorn",
    ];

    let label_font_size = 11.0;
    for (i, &(px, pz)) in PROVINCE_POSITIONS.iter().enumerate() {
        let name = if i < province_names.len() { province_names[i] } else { "?" };
        let label_y = province_elevation(i) + 0.15;
        commands.spawn((
            ProvinceLabel { province_id: i as u16 },
            Text2d::new(name),
            TextFont {
                font_size: label_font_size,
                ..default()
            },
            TextColor(Color::srgba(0.9, 0.9, 0.9, 0.85)),
            // Labels face the tilted camera (atan2(20, 12) ≈ 59° from horizontal).
            Transform::from_xyz(px, label_y, pz - 0.7)
                .with_rotation(Quat::from_rotation_x(-1.03)),
        ));
    }

    // Selection ring — a wireframe circle that follows the selected province.
    // Starts invisible; the `update_selection_ring` system positions it.
    let ring_mesh = meshes.add(build_ring_mesh(HEX_RADIUS + 0.12, 24));
    let ring_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 1.0, 1.0, 0.9),
        unlit: true,
        ..default()
    });
    commands.spawn((
        SelectionRing,
        Mesh3d(ring_mesh),
        MeshMaterial3d(ring_mat),
        Transform::from_xyz(0.0, -100.0, 0.0), // hidden off-screen initially
    ));
}

// ---------------------------------------------------------------------------
// Update system — re-colour provinces when world state changes
// ---------------------------------------------------------------------------

pub fn update_map_colors(
    game_state: Res<ClientGameState>,
    selection: Res<Selection>,
    time: Res<Time>,
    query: Query<(&ProvinceMarker, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Some(ref world) = game_state.world else {
        return;
    };

    // Pre-compute which provinces are under active siege.
    let mut besieged: std::collections::HashSet<u16> = std::collections::HashSet::new();
    for army in &world.armies {
        if let Some(ref siege) = army.siege {
            besieged.insert(siege.target_province);
        }
    }

    // Pulsing effect for sieges (0.3..0.7 sinusoidal).
    let siege_pulse = 0.5 + 0.2 * (time.elapsed_secs() * 2.5).sin();

    for (marker, mat_handle) in query.iter() {
        let pid = marker.province_id as usize;
        if pid >= world.provinces.len() {
            continue;
        }

        let province = &world.provinces[pid];
        let controller = province.controller as usize;

        let faction_rgb = if controller < world.factions.len() {
            world.factions[controller].color_rgb
        } else if controller < DEFAULT_FACTION_COLORS.len() {
            DEFAULT_FACTION_COLORS[controller]
        } else {
            [128, 128, 128]
        };

        let terrain = if pid < DEFAULT_TERRAIN.len() {
            DEFAULT_TERRAIN[pid]
        } else {
            Terrain::Plains
        };

        let mut new_color = province_color(faction_rgb, terrain);

        // Siege indicator — pulsing red tint.
        if besieged.contains(&(pid as u16)) {
            let LinearRgba { red, green, blue, .. } = new_color.to_linear();
            new_color = Color::srgb(
                (red * 0.5 + siege_pulse * 0.5).clamp(0.0, 1.0),
                (green * 0.3).clamp(0.0, 1.0),
                (blue * 0.3).clamp(0.0, 1.0),
            );
        }

        // High unrest (>500) — desaturate toward grey.
        let unrest_val = province.unrest.raw() as f32 / 1000.0; // 0.0-1.0
        if unrest_val > 0.5 {
            let desat = (unrest_val - 0.5) * 2.0; // 0.0 at 500, 1.0 at 1000
            let LinearRgba { red, green, blue, .. } = new_color.to_linear();
            let grey = (red + green + blue) / 3.0;
            new_color = Color::srgb(
                (red + (grey - red) * desat * 0.6).clamp(0.0, 1.0),
                (green + (grey - green) * desat * 0.6).clamp(0.0, 1.0),
                (blue + (grey - blue) * desat * 0.6).clamp(0.0, 1.0),
            );
        }

        // Hover highlight — subtle brightening when mouse is over this province.
        let is_hovered = selection.hovered_province == Some(pid as u16);
        if is_hovered {
            let LinearRgba { red, green, blue, .. } = new_color.to_linear();
            new_color = Color::srgb(
                (red * 1.2 + 0.05).clamp(0.0, 1.0),
                (green * 1.2 + 0.05).clamp(0.0, 1.0),
                (blue * 1.2 + 0.05).clamp(0.0, 1.0),
            );
        }

        // Highlight selected province — brighten it significantly.
        let is_selected = selection.province == Some(pid as u16);
        if is_selected {
            let LinearRgba { red, green, blue, .. } = new_color.to_linear();
            new_color = Color::srgb(
                (red * 1.6 + 0.15).clamp(0.0, 1.0),
                (green * 1.6 + 0.15).clamp(0.0, 1.0),
                (blue * 1.6 + 0.15).clamp(0.0, 1.0),
            );
        }

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
        let elev = province_elevation(loc);

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
            Transform::from_xyz(px + ARMY_OFFSET_X, ARMY_Y + elev, pz + ARMY_OFFSET_Z),
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
    game_state: Res<ClientGameState>,
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
    if ray.direction.y.abs() < 1e-6 {
        return;
    }

    let t = -ray.origin.y / ray.direction.y;
    if t < 0.0 {
        return;
    }

    let hit = ray.origin + t * *ray.direction;
    let hit_x = hit.x;
    let hit_z = hit.z;

    // Check army click first — army cubes sit at ARMY_Y offset from province centre.
    // Test if click is near an army icon (smaller hit radius than province hex).
    if let Some(ref world) = game_state.world {
        let mut best_army: Option<(u32, f32)> = None;
        for army in &world.armies {
            let loc = army.location as usize;
            if loc >= PROVINCE_POSITIONS.len() {
                continue;
            }
            let (px, pz) = PROVINCE_POSITIONS[loc];
            let ax = px + ARMY_OFFSET_X;
            let az = pz + ARMY_OFFSET_Z;
            let dx = hit_x - ax;
            let dz = hit_z - az;
            let dist_sq = dx * dx + dz * dz;
            // Army cube is 0.4 wide — use ~0.5 radius for click detection.
            if dist_sq < 0.25 {
                if best_army.map_or(true, |(_, bd)| dist_sq < bd) {
                    best_army = Some((army.id, dist_sq));
                }
            }
        }

        if let Some((army_id, _)) = best_army {
            selection.army = Some(army_id);
            // Also select the army's province and faction.
            if let Some(army) = world.armies.iter().find(|a| a.id == army_id) {
                selection.province = Some(army.location);
                selection.faction = Some(army.owner_faction);
            }
            return;
        }
    }

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
        selection.army = None; // Clear army selection when clicking a province.

        // Auto-select the controlling faction.
        if let Some(ref world) = game_state.world {
            if let Some(prov) = world.provinces.iter().find(|p| p.id == pid) {
                selection.faction = Some(prov.controller);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Update system — detect hovered province for tooltips
// ---------------------------------------------------------------------------

pub fn update_hover(
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<MapCamera>>,
    mut selection: ResMut<Selection>,
) {
    let Ok(window) = windows.get_single() else {
        selection.hovered_province = None;
        return;
    };
    let Some(cursor_pos) = window.cursor_position() else {
        selection.hovered_province = None;
        return;
    };
    let Ok((camera, cam_transform)) = camera_q.get_single() else {
        selection.hovered_province = None;
        return;
    };

    selection.cursor_screen_pos = Some((cursor_pos.x, cursor_pos.y));

    let Ok(ray) = camera.viewport_to_world(cam_transform, cursor_pos) else {
        selection.hovered_province = None;
        return;
    };

    if ray.direction.y.abs() < 1e-6 {
        selection.hovered_province = None;
        return;
    }

    let t = -ray.origin.y / ray.direction.y;
    if t < 0.0 {
        selection.hovered_province = None;
        return;
    }

    let hit = ray.origin + t * *ray.direction;

    let mut best: Option<(u16, f32)> = None;
    for (i, &(px, pz)) in PROVINCE_POSITIONS.iter().enumerate() {
        let dx = hit.x - px;
        let dz = hit.z - pz;
        let dist_sq = dx * dx + dz * dz;
        if dist_sq < HEX_RADIUS * HEX_RADIUS {
            if best.map_or(true, |(_, bd)| dist_sq < bd) {
                best = Some((i as u16, dist_sq));
            }
        }
    }

    selection.hovered_province = best.map(|(pid, _)| pid);
}

// ---------------------------------------------------------------------------
// Update system — move selection ring to the selected province
// ---------------------------------------------------------------------------

pub fn update_selection_ring(
    selection: Res<Selection>,
    mut query: Query<&mut Transform, With<SelectionRing>>,
) {
    let Ok(mut tf) = query.get_single_mut() else {
        return;
    };

    match selection.province {
        Some(pid) => {
            let idx = pid as usize;
            if idx < PROVINCE_POSITIONS.len() {
                let (px, pz) = PROVINCE_POSITIONS[idx];
                let elev = province_elevation(idx);
                tf.translation = Vec3::new(px, elev + 0.05, pz);
            }
        }
        None => {
            // Hide off-screen when nothing is selected.
            tf.translation.y = -100.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Update system — update province labels from live world data
// ---------------------------------------------------------------------------

pub fn update_province_labels(
    game_state: Res<ClientGameState>,
    mut query: Query<(&ProvinceLabel, &mut Text2d)>,
) {
    let Some(ref world) = game_state.world else {
        return;
    };

    for (label, mut text) in query.iter_mut() {
        if let Some(prov) = world.provinces.iter().find(|p| p.id == label.province_id) {
            // Show name + population as a compact label.
            let pop = if prov.population >= 1000 {
                format!("{:.1}K", prov.population as f64 / 1000.0)
            } else {
                format!("{}", prov.population)
            };
            **text = format!("{}\n{}", prov.name, pop);
        }
    }
}

// ---------------------------------------------------------------------------
// Update system — draw army movement path lines
// ---------------------------------------------------------------------------

/// Height of movement path lines above the hex surface.
const PATH_Y: f32 = 0.08;

pub fn update_movement_paths(
    mut commands: Commands,
    game_state: Res<ClientGameState>,
    existing: Query<(Entity, &MovementPath)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Some(ref world) = game_state.world else {
        // No world data — despawn all movement paths.
        for (entity, _) in existing.iter() {
            commands.entity(entity).despawn();
        }
        return;
    };

    // Build set of armies that are currently moving.
    let moving_armies: std::collections::HashMap<u32, &crown_ash_types::Army> = world
        .armies
        .iter()
        .filter(|a| a.destination.is_some() || !a.movement_queue.is_empty())
        .map(|a| (a.id, a))
        .collect();

    // Despawn path entities for armies that are no longer moving or don't exist.
    for (entity, mp) in existing.iter() {
        if !moving_armies.contains_key(&mp.army_id) {
            commands.entity(entity).despawn();
        }
    }

    // Collect IDs of path entities we already have.
    let existing_ids: std::collections::HashSet<u32> =
        existing.iter().map(|(_, m)| m.army_id).collect();

    // Spawn/update movement path lines.
    for (&army_id, army) in &moving_armies {
        if existing_ids.contains(&army_id) {
            // TODO: could update mesh if path changed, but despawn/respawn is simpler
            // for now since paths change infrequently (once per turn).
            continue;
        }

        // Build the waypoint chain: current location -> destination/queue steps.
        let mut waypoints: Vec<u16> = vec![army.location];
        if !army.movement_queue.is_empty() {
            waypoints.extend_from_slice(&army.movement_queue);
        } else if let Some(dest) = army.destination {
            waypoints.push(dest);
        }

        if waypoints.len() < 2 {
            continue;
        }

        // Build a dashed line mesh along the waypoint chain.
        let mesh = build_path_mesh(&waypoints);
        let mesh_handle = meshes.add(mesh);

        // Resolve faction colour for the path line (slightly brighter/translucent).
        let owner = army.owner_faction as usize;
        let rgb = if owner < world.factions.len() {
            world.factions[owner].color_rgb
        } else if owner < DEFAULT_FACTION_COLORS.len() {
            DEFAULT_FACTION_COLORS[owner]
        } else {
            [200, 200, 200]
        };

        let path_color = Color::srgba(
            (rgb[0] as f32 / 255.0 * 1.4).min(1.0),
            (rgb[1] as f32 / 255.0 * 1.4).min(1.0),
            (rgb[2] as f32 / 255.0 * 1.4).min(1.0),
            0.8,
        );

        let mat = materials.add(StandardMaterial {
            base_color: path_color,
            unlit: true,
            ..default()
        });

        commands.spawn((
            MovementPath { army_id },
            Mesh3d(mesh_handle),
            MeshMaterial3d(mat),
            Transform::IDENTITY,
        ));
    }
}

/// Build a dashed line-list mesh following the given province waypoints.
/// Each segment between waypoints is drawn as dashes (short line segments).
fn build_path_mesh(waypoints: &[u16]) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();

    let dash_len = 0.3_f32;
    let gap_len = 0.15_f32;

    for pair in waypoints.windows(2) {
        let from_idx = pair[0] as usize;
        let to_idx = pair[1] as usize;
        if from_idx >= PROVINCE_POSITIONS.len() || to_idx >= PROVINCE_POSITIONS.len() {
            continue;
        }

        let (fx, fz) = PROVINCE_POSITIONS[from_idx];
        let (tx, tz) = PROVINCE_POSITIONS[to_idx];
        let fy = province_elevation(from_idx) + PATH_Y;
        let ty = province_elevation(to_idx) + PATH_Y;

        let dx = tx - fx;
        let dz = tz - fz;
        let total_len = (dx * dx + dz * dz).sqrt();
        if total_len < 0.01 {
            continue;
        }

        let nx = dx / total_len;
        let nz = dz / total_len;

        // Walk along the segment, emitting dashes with interpolated elevation.
        let mut t = 0.0_f32;
        while t < total_len {
            let dash_end = (t + dash_len).min(total_len);
            let frac0 = t / total_len;
            let frac1 = dash_end / total_len;
            let x0 = fx + nx * t;
            let z0 = fz + nz * t;
            let y0 = fy + (ty - fy) * frac0;
            let x1 = fx + nx * dash_end;
            let z1 = fz + nz * dash_end;
            let y1 = fy + (ty - fy) * frac1;

            positions.push([x0, y0, z0]);
            positions.push([x1, y1, z1]);
            normals.push([0.0, 1.0, 0.0]);
            normals.push([0.0, 1.0, 0.0]);

            t = dash_end + gap_len;
        }
    }

    // Add arrowhead at the final destination.
    if waypoints.len() >= 2 {
        let last = *waypoints.last().unwrap() as usize;
        let prev = waypoints[waypoints.len() - 2] as usize;
        if last < PROVINCE_POSITIONS.len() && prev < PROVINCE_POSITIONS.len() {
            let (tx, tz) = PROVINCE_POSITIONS[last];
            let (fx, fz) = PROVINCE_POSITIONS[prev];
            let arrow_y = province_elevation(last) + PATH_Y;
            let dx = tx - fx;
            let dz = tz - fz;
            let len = (dx * dx + dz * dz).sqrt();
            if len > 0.01 {
                let nx = dx / len;
                let nz = dz / len;
                let arrow_size = 0.3;
                let tip_x = tx - nx * 0.2;
                let tip_z = tz - nz * 0.2;
                let base_x = tip_x - nx * arrow_size;
                let base_z = tip_z - nz * arrow_size;
                let px = -nz * arrow_size * 0.5;
                let pz = nx * arrow_size * 0.5;

                positions.push([base_x + px, arrow_y, base_z + pz]);
                positions.push([tip_x, arrow_y, tip_z]);
                normals.push([0.0, 1.0, 0.0]);
                normals.push([0.0, 1.0, 0.0]);

                positions.push([base_x - px, arrow_y, base_z - pz]);
                positions.push([tip_x, arrow_y, tip_z]);
                normals.push([0.0, 1.0, 0.0]);
                normals.push([0.0, 1.0, 0.0]);
            }
        }
    }

    Mesh::new(
        PrimitiveTopology::LineList,
        bevy::render::render_asset::RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
}

// ---------------------------------------------------------------------------
// Update system — smoothly interpolate army positions toward destinations
// ---------------------------------------------------------------------------

pub fn animate_army_movement(
    game_state: Res<ClientGameState>,
    time: Res<Time>,
    mut query: Query<(&ArmyMarker, &mut Transform)>,
) {
    let Some(ref world) = game_state.world else {
        return;
    };

    let speed = 2.0_f32; // units per second interpolation speed

    for (marker, mut tf) in query.iter_mut() {
        let Some(army) = world.armies.iter().find(|a| a.id == marker.army_id) else {
            continue;
        };

        // Determine target position — if moving, lerp toward destination.
        let target_loc = if !army.movement_queue.is_empty() {
            army.movement_queue[0] as usize
        } else if let Some(dest) = army.destination {
            dest as usize
        } else {
            army.location as usize
        };

        if target_loc >= PROVINCE_POSITIONS.len() {
            continue;
        }

        let (px, pz) = PROVINCE_POSITIONS[target_loc];
        let target_elev = province_elevation(target_loc);
        let target = Vec3::new(px + ARMY_OFFSET_X, ARMY_Y + target_elev, pz + ARMY_OFFSET_Z);

        // Smoothly move toward target.
        let current = tf.translation;
        let diff = target - current;
        let dist = diff.length();
        if dist > 0.05 {
            let step = (speed * time.delta_secs()).min(dist);
            tf.translation += diff.normalize() * step;
        } else {
            tf.translation = target;
        }

        // Add a slight bobbing effect for armies in transit.
        if army.destination.is_some() || !army.movement_queue.is_empty() {
            let bob = (time.elapsed_secs() * 3.0 + marker.army_id as f32).sin() * 0.05;
            tf.translation.y = ARMY_Y + target_elev + bob;
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh builder: selection ring (line-loop circle)
// ---------------------------------------------------------------------------

fn build_ring_mesh(radius: f32, segments: u32) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity((segments * 2) as usize);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity((segments * 2) as usize);

    for i in 0..segments {
        let a0 = (i as f32) * 2.0 * PI / (segments as f32);
        let a1 = ((i + 1) as f32) * 2.0 * PI / (segments as f32);
        positions.push([radius * a0.cos(), 0.0, radius * a0.sin()]);
        positions.push([radius * a1.cos(), 0.0, radius * a1.sin()]);
        normals.push([0.0, 1.0, 0.0]);
        normals.push([0.0, 1.0, 0.0]);
    }

    Mesh::new(PrimitiveTopology::LineList, bevy::render::render_asset::RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
}

// ---------------------------------------------------------------------------
// Siege & battle visual markers
// ---------------------------------------------------------------------------

/// Tags a floating marker above a besieged province.
#[derive(Component)]
pub struct SiegeMarker {
    pub province_id: u16,
}

/// Spawns/despawns floating siege markers above besieged provinces.
/// Crossed-sword icons bob and rotate to draw attention.
pub fn update_siege_markers(
    mut commands: Commands,
    game_state: Res<ClientGameState>,
    existing: Query<(Entity, &SiegeMarker)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Some(ref world) = game_state.world else {
        for (entity, _) in existing.iter() {
            commands.entity(entity).despawn();
        }
        return;
    };

    // Collect currently besieged provinces.
    let mut besieged: std::collections::HashSet<u16> = std::collections::HashSet::new();
    for army in &world.armies {
        if let Some(ref siege) = army.siege {
            besieged.insert(siege.target_province);
        }
    }

    // Despawn markers for provinces no longer besieged; animate those that remain.
    let mut existing_ids: std::collections::HashSet<u16> = std::collections::HashSet::new();
    for (entity, marker) in existing.iter() {
        if !besieged.contains(&marker.province_id) {
            commands.entity(entity).despawn();
        } else {
            existing_ids.insert(marker.province_id);
        }
    }

    // Spawn new markers for newly besieged provinces.
    for &pid16 in &besieged {
        if existing_ids.contains(&pid16) {
            continue;
        }
        let pid = pid16 as usize;
        if pid >= PROVINCE_POSITIONS.len() {
            continue;
        }

        let (px, pz) = PROVINCE_POSITIONS[pid];
        let elev = province_elevation(pid);

        // Crossed-swords mesh: two thin cuboids at ±45° forming an X.
        let sword = meshes.add(Cuboid::new(0.05, 0.4, 0.05));
        let mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.15, 0.1),
            emissive: LinearRgba::new(2.0, 0.3, 0.1, 1.0),
            ..default()
        });

        commands.spawn((
            SiegeMarker { province_id: pid16 },
            Mesh3d(sword.clone()),
            MeshMaterial3d(mat.clone()),
            Transform::from_xyz(px - 0.1, elev + 1.2, pz)
                .with_rotation(Quat::from_rotation_z(0.7)),
        ));

        commands.spawn((
            SiegeMarker { province_id: pid16 },
            Mesh3d(sword),
            MeshMaterial3d(mat),
            Transform::from_xyz(px + 0.1, elev + 1.2, pz)
                .with_rotation(Quat::from_rotation_z(-0.7)),
        ));
    }
}

/// Animates siege markers — bobbing and slow rotation.
pub fn animate_siege_markers(
    time: Res<Time>,
    mut query: Query<(&SiegeMarker, &mut Transform)>,
) {
    for (marker, mut tf) in query.iter_mut() {
        let pid = marker.province_id as usize;
        if pid >= PROVINCE_POSITIONS.len() { continue; }
        let elev = province_elevation(pid);
        let bob = (time.elapsed_secs() * 2.0 + marker.province_id as f32 * 0.5).sin() * 0.15;
        tf.translation.y = elev + 1.2 + bob;
    }
}
