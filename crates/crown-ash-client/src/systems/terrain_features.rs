//! Procedural terrain feature system — generates trees, rocks, water, and
//! sand dune meshes based on province terrain type.
//!
//! All meshes are generated at startup and instanced per province.
//! Features are scattered within each hex using deterministic pseudo-random
//! positions seeded by province ID.

use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use std::f32::consts::PI;

use crate::systems::map_render::{PROVINCE_POSITIONS, province_elevation};

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Tags a terrain feature entity so we can query/despawn them.
#[derive(Component)]
pub struct TerrainFeature {
    pub province_id: u16,
}

/// Tags the animated water overlay for Coastal/River/Marsh provinces.
#[derive(Component)]
pub struct WaterPlane {
    pub province_id: u16,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const HEX_RADIUS: f32 = 1.35;

/// Terrain type IDs matching map_render.rs order.
const TERRAIN_PLAINS: u8 = 0;
const TERRAIN_HILLS: u8 = 1;
const TERRAIN_MOUNTAINS: u8 = 2;
const TERRAIN_FOREST: u8 = 3;
const TERRAIN_MARSH: u8 = 4;
const TERRAIN_DESERT: u8 = 5;
const TERRAIN_COASTAL: u8 = 6;
const TERRAIN_RIVER: u8 = 7;

/// Default terrain per province (mirrors map_render.rs).
const DEFAULT_TERRAIN: [u8; 25] = [
    TERRAIN_MOUNTAINS, TERRAIN_HILLS, TERRAIN_MARSH, TERRAIN_COASTAL,
    TERRAIN_PLAINS, TERRAIN_HILLS, TERRAIN_FOREST,
    TERRAIN_PLAINS, TERRAIN_HILLS, TERRAIN_RIVER, TERRAIN_PLAINS,
    TERRAIN_HILLS, TERRAIN_PLAINS, TERRAIN_FOREST,
    TERRAIN_COASTAL, TERRAIN_COASTAL, TERRAIN_COASTAL, TERRAIN_PLAINS,
    TERRAIN_FOREST, TERRAIN_HILLS, TERRAIN_MOUNTAINS,
    TERRAIN_PLAINS, TERRAIN_DESERT, TERRAIN_DESERT, TERRAIN_PLAINS,
];

// ---------------------------------------------------------------------------
// Deterministic pseudo-random scatter
// ---------------------------------------------------------------------------

/// Simple hash for deterministic placement. Returns 0.0..1.0.
fn hash_f32(seed: u32) -> f32 {
    let h = seed.wrapping_mul(2654435761);
    let h = h ^ (h >> 16);
    (h & 0xFFFF) as f32 / 65535.0
}

/// Returns a point inside a hex (rejection sampling with deterministic hash).
/// Returns (x_offset, z_offset) relative to hex centre.
fn scatter_in_hex(province_id: u16, index: u32, radius: f32) -> (f32, f32) {
    // Use multiple hash rounds to get a point inside the hex.
    let seed = (province_id as u32) * 1000 + index * 7 + 42;
    let angle = hash_f32(seed) * 2.0 * PI;
    let r = hash_f32(seed + 1) * radius * 0.85; // stay inside hex with margin
    (r * angle.cos(), r * angle.sin())
}

// ---------------------------------------------------------------------------
// Procedural mesh builders
// ---------------------------------------------------------------------------

/// Builds a simple tree mesh: cylinder trunk + cone canopy.
/// Total height ~0.8 units, centred at origin.
fn build_tree_mesh(segments: u32) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let trunk_radius = 0.04;
    let trunk_height = 0.25;
    let canopy_radius = 0.2;
    let canopy_height = 0.55;

    // --- Trunk (cylinder) ---
    let base_idx = positions.len() as u32;
    // Bottom centre
    positions.push([0.0, 0.0, 0.0]);
    normals.push([0.0, -1.0, 0.0]);
    uvs.push([0.5, 0.5]);
    // Top centre
    positions.push([0.0, trunk_height, 0.0]);
    normals.push([0.0, 1.0, 0.0]);
    uvs.push([0.5, 0.5]);

    for i in 0..segments {
        let angle = (i as f32) * 2.0 * PI / segments as f32;
        let x = trunk_radius * angle.cos();
        let z = trunk_radius * angle.sin();
        let nx = angle.cos();
        let nz = angle.sin();

        // Bottom ring vertex
        positions.push([x, 0.0, z]);
        normals.push([nx, 0.0, nz]);
        uvs.push([i as f32 / segments as f32, 0.0]);
        // Top ring vertex
        positions.push([x, trunk_height, z]);
        normals.push([nx, 0.0, nz]);
        uvs.push([i as f32 / segments as f32, 1.0]);
    }

    // Trunk side faces
    for i in 0..segments {
        let next = (i + 1) % segments;
        let b0 = base_idx + 2 + i * 2;
        let t0 = base_idx + 2 + i * 2 + 1;
        let b1 = base_idx + 2 + next * 2;
        let t1 = base_idx + 2 + next * 2 + 1;
        indices.extend_from_slice(&[b0, t0, b1, b1, t0, t1]);
    }

    // --- Canopy (cone) ---
    let cone_base_y = trunk_height;
    let cone_tip_y = trunk_height + canopy_height;
    let cone_base_idx = positions.len() as u32;

    // Cone tip
    positions.push([0.0, cone_tip_y, 0.0]);
    normals.push([0.0, 1.0, 0.0]);
    uvs.push([0.5, 1.0]);

    // Cone base centre (for bottom cap)
    positions.push([0.0, cone_base_y, 0.0]);
    normals.push([0.0, -1.0, 0.0]);
    uvs.push([0.5, 0.0]);

    for i in 0..segments {
        let angle = (i as f32) * 2.0 * PI / segments as f32;
        let x = canopy_radius * angle.cos();
        let z = canopy_radius * angle.sin();
        // Normal points outward and slightly up for a cone.
        let slope = canopy_radius / canopy_height;
        let ny = slope;
        let len = (1.0 + ny * ny).sqrt();
        positions.push([x, cone_base_y, z]);
        normals.push([angle.cos() / len, ny / len, angle.sin() / len]);
        uvs.push([i as f32 / segments as f32, 0.0]);
    }

    // Cone side triangles (tip to base ring)
    let tip_idx = cone_base_idx;
    for i in 0..segments {
        let next = (i + 1) % segments;
        let v0 = cone_base_idx + 2 + i;
        let v1 = cone_base_idx + 2 + next;
        indices.extend_from_slice(&[tip_idx, v1, v0]);
    }

    // Cone bottom cap
    let cap_centre = cone_base_idx + 1;
    for i in 0..segments {
        let next = (i + 1) % segments;
        let v0 = cone_base_idx + 2 + i;
        let v1 = cone_base_idx + 2 + next;
        indices.extend_from_slice(&[cap_centre, v0, v1]);
    }

    Mesh::new(PrimitiveTopology::TriangleList, bevy::render::render_asset::RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
}

/// Builds a rough rock mesh (displaced octahedron for organic look).
fn build_rock_mesh(seed: u32) -> Mesh {
    // Start with a low-poly sphere (icosahedron-like) and displace vertices.
    let base_verts: Vec<[f32; 3]> = vec![
        [ 0.0,  0.5,  0.0],  // top
        [ 0.5,  0.0,  0.0],  // right
        [ 0.0,  0.0,  0.5],  // front
        [-0.5,  0.0,  0.0],  // left
        [ 0.0,  0.0, -0.5],  // back
        [ 0.0, -0.2,  0.0],  // bottom (flatter)
    ];

    let base_tris: Vec<[u32; 3]> = vec![
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], // top half
        [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4], // bottom half
    ];

    // Displace each vertex slightly for organic variation.
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for (ti, tri) in base_tris.iter().enumerate() {
        let v0 = base_verts[tri[0] as usize];
        let v1 = base_verts[tri[1] as usize];
        let v2 = base_verts[tri[2] as usize];

        // Displace vertices
        let displace = |v: [f32; 3], vi: u32| -> [f32; 3] {
            let d = (hash_f32(seed + vi * 37 + ti as u32 * 13) - 0.5) * 0.15;
            [v[0] + d, (v[1] + d * 0.5).max(-0.1), v[2] + d]
        };

        let p0 = displace(v0, tri[0]);
        let p1 = displace(v1, tri[1]);
        let p2 = displace(v2, tri[2]);

        // Compute face normal.
        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(0.001);
        let normal = [n[0] / len, n[1] / len, n[2] / len];

        let base = positions.len() as u32;
        positions.push(p0);
        positions.push(p1);
        positions.push(p2);
        normals.push(normal);
        normals.push(normal);
        normals.push(normal);
        uvs.push([0.0, 0.0]);
        uvs.push([1.0, 0.0]);
        uvs.push([0.5, 1.0]);
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    Mesh::new(PrimitiveTopology::TriangleList, bevy::render::render_asset::RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
}

/// Builds a flat hexagonal water overlay mesh (slightly smaller than the province hex).
fn build_water_hex(radius: f32) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(7);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(7);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(7);
    let mut indices: Vec<u32> = Vec::with_capacity(18);

    positions.push([0.0, 0.0, 0.0]);
    normals.push([0.0, 1.0, 0.0]);
    uvs.push([0.5, 0.5]);

    for i in 0..6 {
        let angle = (i as f32) * PI / 3.0;
        let x = radius * angle.cos();
        let z = radius * angle.sin();
        positions.push([x, 0.0, z]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([0.5 + 0.5 * angle.cos(), 0.5 + 0.5 * angle.sin()]);
    }

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

/// Builds a sand dune bump — elongated dome shape.
fn build_dune_mesh(segments: u32) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let width = 0.4_f32;
    let length = 0.8_f32;
    let height = 0.12_f32;

    // Centre top
    positions.push([0.0, height, 0.0]);
    normals.push([0.0, 1.0, 0.0]);
    uvs.push([0.5, 0.5]);

    // Elliptical base ring
    for i in 0..segments {
        let angle = (i as f32) * 2.0 * PI / segments as f32;
        let x = length * angle.cos();
        let z = width * angle.sin();
        positions.push([x, 0.0, z]);
        let nx = angle.cos() * height / length;
        let nz = angle.sin() * height / width;
        let len = (nx * nx + 1.0 + nz * nz).sqrt();
        normals.push([nx / len, 1.0 / len, nz / len]);
        uvs.push([0.5 + 0.5 * angle.cos(), 0.5 + 0.5 * angle.sin()]);
    }

    for i in 0u32..segments {
        let next = (i + 1) % segments;
        indices.extend_from_slice(&[0, i + 1, next + 1]);
    }

    Mesh::new(PrimitiveTopology::TriangleList, bevy::render::render_asset::RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
}

/// Builds a small grass tuft mesh (3 crossed quads).
fn build_grass_tuft() -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let h = 0.15_f32;
    let w = 0.06_f32;

    // 3 crossed blades at 60° apart
    for blade in 0..3 {
        let angle = (blade as f32) * PI / 3.0;
        let dx = w * angle.cos();
        let dz = w * angle.sin();
        let nx = -angle.sin();
        let nz = angle.cos();
        let base = positions.len() as u32;

        positions.push([-dx, 0.0, -dz]);
        positions.push([ dx, 0.0,  dz]);
        positions.push([ dx * 0.3, h,  dz * 0.3]);
        positions.push([-dx * 0.3, h, -dz * 0.3]);

        for _ in 0..4 {
            normals.push([nx, 0.3, nz]);
        }
        uvs.push([0.0, 0.0]);
        uvs.push([1.0, 0.0]);
        uvs.push([1.0, 1.0]);
        uvs.push([0.0, 1.0]);

        indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
        // Back face
        indices.extend_from_slice(&[base, base+2, base+1, base, base+3, base+2]);
    }

    Mesh::new(PrimitiveTopology::TriangleList, bevy::render::render_asset::RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
}

// ---------------------------------------------------------------------------
// Resource: prebuilt terrain meshes
// ---------------------------------------------------------------------------

/// Holds prebuilt mesh handles for terrain features.
#[derive(Resource)]
pub struct TerrainMeshes {
    pub tree: Handle<Mesh>,
    pub rock_a: Handle<Mesh>,
    pub rock_b: Handle<Mesh>,
    pub rock_c: Handle<Mesh>,
    pub water_hex: Handle<Mesh>,
    pub dune: Handle<Mesh>,
    pub grass: Handle<Mesh>,
    // Materials
    pub tree_trunk_mat: Handle<StandardMaterial>,
    pub tree_canopy_mat: Handle<StandardMaterial>,
    pub rock_mat: Handle<StandardMaterial>,
    pub water_mat: Handle<StandardMaterial>,
    pub dune_mat: Handle<StandardMaterial>,
    pub grass_mat: Handle<StandardMaterial>,
    pub marsh_mat: Handle<StandardMaterial>,
}

// ---------------------------------------------------------------------------
// Startup system
// ---------------------------------------------------------------------------

/// Generates all procedural terrain meshes and materials at startup.
pub fn setup_terrain_features(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let tree = meshes.add(build_tree_mesh(6));
    let rock_a = meshes.add(build_rock_mesh(100));
    let rock_b = meshes.add(build_rock_mesh(200));
    let rock_c = meshes.add(build_rock_mesh(300));
    let water_hex = meshes.add(build_water_hex(HEX_RADIUS * 0.92));
    let dune = meshes.add(build_dune_mesh(8));
    let grass = meshes.add(build_grass_tuft());

    // Materials — lit for 3D depth with the sun light.
    let tree_trunk_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.35, 0.22, 0.10),
        perceptual_roughness: 0.9,
        ..default()
    });
    let tree_canopy_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.20, 0.50, 0.15),
        perceptual_roughness: 0.95,
        ..default()
    });
    let rock_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.45, 0.43, 0.40),
        perceptual_roughness: 0.85,
        ..default()
    });
    let water_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.15, 0.35, 0.65, 0.55),
        alpha_mode: AlphaMode::Blend,
        double_sided: true,
        perceptual_roughness: 0.1,
        metallic: 0.3,
        ..default()
    });
    let dune_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.75, 0.50),
        perceptual_roughness: 0.95,
        ..default()
    });
    let grass_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.30, 0.55, 0.20),
        double_sided: true,
        perceptual_roughness: 0.95,
        ..default()
    });
    let marsh_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.20, 0.38, 0.30, 0.50),
        alpha_mode: AlphaMode::Blend,
        double_sided: true,
        perceptual_roughness: 0.3,
        ..default()
    });

    commands.insert_resource(TerrainMeshes {
        tree, rock_a, rock_b, rock_c, water_hex, dune, grass,
        tree_trunk_mat, tree_canopy_mat, rock_mat, water_mat,
        dune_mat, grass_mat, marsh_mat,
    });
}

/// Spawns terrain features for all 25 provinces based on their terrain type.
///
/// Called once at startup. Features are static — they don't change during play.
pub fn spawn_terrain_features(
    mut commands: Commands,
    terrain_meshes: Res<TerrainMeshes>,
) {
    for (pid, &(px, pz)) in PROVINCE_POSITIONS.iter().enumerate() {
        if pid >= DEFAULT_TERRAIN.len() { break; }
        let terrain = DEFAULT_TERRAIN[pid];
        let elev = province_elevation(pid);

        match terrain {
            TERRAIN_FOREST => {
                spawn_forest(&mut commands, &terrain_meshes, pid as u16, px, pz, elev);
            }
            TERRAIN_MOUNTAINS => {
                spawn_mountains(&mut commands, &terrain_meshes, pid as u16, px, pz, elev);
            }
            TERRAIN_COASTAL | TERRAIN_RIVER => {
                spawn_water(&mut commands, &terrain_meshes, pid as u16, px, pz, elev, false);
            }
            TERRAIN_MARSH => {
                spawn_water(&mut commands, &terrain_meshes, pid as u16, px, pz, elev, true);
                spawn_marsh_reeds(&mut commands, &terrain_meshes, pid as u16, px, pz, elev);
            }
            TERRAIN_DESERT => {
                spawn_dunes(&mut commands, &terrain_meshes, pid as u16, px, pz, elev);
            }
            TERRAIN_HILLS => {
                spawn_hills_grass(&mut commands, &terrain_meshes, pid as u16, px, pz, elev);
            }
            TERRAIN_PLAINS => {
                spawn_plains_grass(&mut commands, &terrain_meshes, pid as u16, px, pz, elev);
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Terrain-specific spawners
// ---------------------------------------------------------------------------

fn spawn_forest(
    commands: &mut Commands,
    tm: &TerrainMeshes,
    pid: u16, px: f32, pz: f32, elev: f32,
) {
    // 5-8 trees scattered in the hex.
    let count = 5 + ((pid as usize * 3) % 4); // 5-8
    for i in 0..count {
        let (ox, oz) = scatter_in_hex(pid, i as u32, HEX_RADIUS);
        let scale = 0.6 + hash_f32(pid as u32 * 100 + i as u32) * 0.6; // 0.6-1.2
        let y_rot = hash_f32(pid as u32 * 200 + i as u32) * 2.0 * PI;

        // Tree uses the combined mesh (trunk + canopy in one), coloured green.
        commands.spawn((
            TerrainFeature { province_id: pid },
            Mesh3d(tm.tree.clone()),
            MeshMaterial3d(tm.tree_canopy_mat.clone()),
            Transform {
                translation: Vec3::new(px + ox, elev, pz + oz),
                rotation: Quat::from_rotation_y(y_rot),
                scale: Vec3::splat(scale),
            },
        ));
    }
}

fn spawn_mountains(
    commands: &mut Commands,
    tm: &TerrainMeshes,
    pid: u16, px: f32, pz: f32, elev: f32,
) {
    // 3-5 rocks of varying size.
    let count = 3 + ((pid as usize * 5) % 3); // 3-5
    for i in 0..count {
        let (ox, oz) = scatter_in_hex(pid, i as u32, HEX_RADIUS * 0.7);
        let scale = 0.3 + hash_f32(pid as u32 * 300 + i as u32) * 0.5; // 0.3-0.8
        let y_rot = hash_f32(pid as u32 * 400 + i as u32) * 2.0 * PI;

        let rock = match i % 3 {
            0 => tm.rock_a.clone(),
            1 => tm.rock_b.clone(),
            _ => tm.rock_c.clone(),
        };

        commands.spawn((
            TerrainFeature { province_id: pid },
            Mesh3d(rock),
            MeshMaterial3d(tm.rock_mat.clone()),
            Transform {
                translation: Vec3::new(px + ox, elev, pz + oz),
                rotation: Quat::from_rotation_y(y_rot),
                scale: Vec3::splat(scale),
            },
        ));
    }
}

fn spawn_water(
    commands: &mut Commands,
    tm: &TerrainMeshes,
    pid: u16, px: f32, pz: f32, elev: f32,
    is_marsh: bool,
) {
    let mat = if is_marsh { tm.marsh_mat.clone() } else { tm.water_mat.clone() };

    commands.spawn((
        WaterPlane { province_id: pid },
        Mesh3d(tm.water_hex.clone()),
        MeshMaterial3d(mat),
        Transform::from_xyz(px, elev + 0.03, pz), // slightly above hex surface
    ));
}

fn spawn_marsh_reeds(
    commands: &mut Commands,
    tm: &TerrainMeshes,
    pid: u16, px: f32, pz: f32, elev: f32,
) {
    // 4-6 grass tufts to look like reeds.
    let count = 4 + ((pid as usize * 7) % 3);
    for i in 0..count {
        let (ox, oz) = scatter_in_hex(pid, i as u32 + 20, HEX_RADIUS * 0.8);
        let scale = 1.0 + hash_f32(pid as u32 * 500 + i as u32) * 0.8;

        commands.spawn((
            TerrainFeature { province_id: pid },
            Mesh3d(tm.grass.clone()),
            MeshMaterial3d(tm.grass_mat.clone()),
            Transform {
                translation: Vec3::new(px + ox, elev + 0.02, pz + oz),
                rotation: Quat::from_rotation_y(hash_f32(pid as u32 * 600 + i as u32) * PI),
                scale: Vec3::splat(scale),
            },
        ));
    }
}

fn spawn_dunes(
    commands: &mut Commands,
    tm: &TerrainMeshes,
    pid: u16, px: f32, pz: f32, elev: f32,
) {
    // 2-4 sand dunes.
    let count = 2 + ((pid as usize * 11) % 3);
    for i in 0..count {
        let (ox, oz) = scatter_in_hex(pid, i as u32 + 30, HEX_RADIUS * 0.7);
        let scale = 0.8 + hash_f32(pid as u32 * 700 + i as u32) * 0.6;
        let y_rot = hash_f32(pid as u32 * 800 + i as u32) * PI; // random orientation

        commands.spawn((
            TerrainFeature { province_id: pid },
            Mesh3d(tm.dune.clone()),
            MeshMaterial3d(tm.dune_mat.clone()),
            Transform {
                translation: Vec3::new(px + ox, elev + 0.01, pz + oz),
                rotation: Quat::from_rotation_y(y_rot),
                scale: Vec3::splat(scale),
            },
        ));
    }
}

fn spawn_hills_grass(
    commands: &mut Commands,
    tm: &TerrainMeshes,
    pid: u16, px: f32, pz: f32, elev: f32,
) {
    // 3-5 grass tufts on hills.
    let count = 3 + ((pid as usize * 13) % 3);
    for i in 0..count {
        let (ox, oz) = scatter_in_hex(pid, i as u32 + 40, HEX_RADIUS * 0.8);
        let scale = 0.7 + hash_f32(pid as u32 * 900 + i as u32) * 0.5;

        commands.spawn((
            TerrainFeature { province_id: pid },
            Mesh3d(tm.grass.clone()),
            MeshMaterial3d(tm.grass_mat.clone()),
            Transform {
                translation: Vec3::new(px + ox, elev + 0.01, pz + oz),
                rotation: Quat::from_rotation_y(hash_f32(pid as u32 * 1000 + i as u32) * PI * 2.0),
                scale: Vec3::splat(scale),
            },
        ));
    }

    // Occasional single tree on hills.
    if pid % 3 == 0 {
        let (ox, oz) = scatter_in_hex(pid, 50, HEX_RADIUS * 0.5);
        commands.spawn((
            TerrainFeature { province_id: pid },
            Mesh3d(tm.tree.clone()),
            MeshMaterial3d(tm.tree_canopy_mat.clone()),
            Transform {
                translation: Vec3::new(px + ox, elev, pz + oz),
                rotation: Quat::from_rotation_y(hash_f32(pid as u32 * 1100) * PI),
                scale: Vec3::splat(0.5),
            },
        ));
    }
}

fn spawn_plains_grass(
    commands: &mut Commands,
    tm: &TerrainMeshes,
    pid: u16, px: f32, pz: f32, elev: f32,
) {
    // 2-3 sparse grass tufts on plains.
    let count = 2 + ((pid as usize * 17) % 2);
    for i in 0..count {
        let (ox, oz) = scatter_in_hex(pid, i as u32 + 60, HEX_RADIUS * 0.9);
        let scale = 0.5 + hash_f32(pid as u32 * 1200 + i as u32) * 0.4;

        commands.spawn((
            TerrainFeature { province_id: pid },
            Mesh3d(tm.grass.clone()),
            MeshMaterial3d(tm.grass_mat.clone()),
            Transform {
                translation: Vec3::new(px + ox, elev + 0.01, pz + oz),
                rotation: Quat::from_rotation_y(hash_f32(pid as u32 * 1300 + i as u32) * PI * 2.0),
                scale: Vec3::splat(scale),
            },
        ));
    }
}

// ---------------------------------------------------------------------------
// Update system — animate water planes (gentle bob)
// ---------------------------------------------------------------------------

/// Slowly bobs water planes up and down for a subtle wave effect.
pub fn animate_water(
    time: Res<Time>,
    mut query: Query<(&WaterPlane, &mut Transform)>,
) {
    let t = time.elapsed_secs();
    for (water, mut tf) in query.iter_mut() {
        let pid = water.province_id;
        let elev = province_elevation(pid as usize);
        // Gentle sine wave: ±0.02 units, different phase per province.
        let wave = (t * 0.8 + pid as f32 * 1.3).sin() * 0.02;
        tf.translation.y = elev + 0.03 + wave;
    }
}
