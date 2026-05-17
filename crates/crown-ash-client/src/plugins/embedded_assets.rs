//! Embeds all GLB model assets directly into the binary so users don't need
//! a separate `assets/` folder.
//!
//! Uses `include_bytes!` to bake each GLB file into the binary at compile time,
//! then registers them with Bevy's `EmbeddedAssetRegistry` at runtime.
//! Building and decoration systems load them via the `embedded://` prefix.

use bevy::prelude::*;
use bevy::asset::io::embedded::EmbeddedAssetRegistry;
use std::path::{Path, PathBuf};

/// Prefix for all embedded asset paths.
/// Usage: `format!("{EMBEDDED_PREFIX}models/village/barn.glb#Scene0")`
pub const EMBEDDED_PREFIX: &str = "embedded://crown_ash_client/";

/// Plugin that registers all 27 GLB model files as embedded assets.
pub struct EmbeddedAssetsPlugin;

impl Plugin for EmbeddedAssetsPlugin {
    fn build(&self, app: &mut App) {
        // (path_in_embedded_fs, bytes)
        let assets: &[(&str, &[u8])] = &[
            // ── Village models (12) ──
            ("models/village/barn.glb", include_bytes!("../../assets/models/village/barn.glb")),
            ("models/village/shed.glb", include_bytes!("../../assets/models/village/shed.glb")),
            ("models/village/woodpile.glb", include_bytes!("../../assets/models/village/woodpile.glb")),
            ("models/village/longhouse.glb", include_bytes!("../../assets/models/village/longhouse.glb")),
            ("models/village/granary.glb", include_bytes!("../../assets/models/village/granary.glb")),
            ("models/village/peasant_house_a.glb", include_bytes!("../../assets/models/village/peasant_house_a.glb")),
            ("models/village/peasant_house_b.glb", include_bytes!("../../assets/models/village/peasant_house_b.glb")),
            ("models/village/peasant_house_c.glb", include_bytes!("../../assets/models/village/peasant_house_c.glb")),
            ("models/village/well.glb", include_bytes!("../../assets/models/village/well.glb")),
            ("models/village/chicken_coop.glb", include_bytes!("../../assets/models/village/chicken_coop.glb")),
            ("models/village/fenced_yard.glb", include_bytes!("../../assets/models/village/fenced_yard.glb")),
            ("models/village/bakehouse.glb", include_bytes!("../../assets/models/village/bakehouse.glb")),
            // ── Town models (12) ──
            ("models/town/stable_block.glb", include_bytes!("../../assets/models/town/stable_block.glb")),
            ("models/town/market_stall_a.glb", include_bytes!("../../assets/models/town/market_stall_a.glb")),
            ("models/town/market_stall_b.glb", include_bytes!("../../assets/models/town/market_stall_b.glb")),
            ("models/town/market_stall_c.glb", include_bytes!("../../assets/models/town/market_stall_c.glb")),
            ("models/town/market_stall_d.glb", include_bytes!("../../assets/models/town/market_stall_d.glb")),
            ("models/town/town_gatehouse.glb", include_bytes!("../../assets/models/town/town_gatehouse.glb")),
            ("models/town/merchant_house.glb", include_bytes!("../../assets/models/town/merchant_house.glb")),
            ("models/town/warehouse.glb", include_bytes!("../../assets/models/town/warehouse.glb")),
            ("models/town/workshop.glb", include_bytes!("../../assets/models/town/workshop.glb")),
            ("models/town/tavern.glb", include_bytes!("../../assets/models/town/tavern.glb")),
            ("models/town/storehouse.glb", include_bytes!("../../assets/models/town/storehouse.glb")),
            ("models/town/blacksmith.glb", include_bytes!("../../assets/models/town/blacksmith.glb")),
            // ── Religious models (3) ──
            ("models/religious/chapel.glb", include_bytes!("../../assets/models/religious/chapel.glb")),
            ("models/religious/monastery.glb", include_bytes!("../../assets/models/religious/monastery.glb")),
            ("models/religious/roadside_shrine.glb", include_bytes!("../../assets/models/religious/roadside_shrine.glb")),
        ];

        // Register each asset in the embedded filesystem.
        // asset_path includes the crate name prefix so that
        // `embedded://crown_ash_client/{rel_path}` resolves correctly.
        let embedded = app.world_mut().resource_mut::<EmbeddedAssetRegistry>();
        for (rel_path, data) in assets {
            let asset_path = Path::new("crown_ash_client").join(rel_path);
            embedded.insert_asset(
                PathBuf::new(),  // full_path (unused without embedded_watcher feature)
                &asset_path,
                *data,
            );
        }
    }
}
