use bevy::prelude::*;
use crown_ash_client::plugins::embedded_assets::EmbeddedAssetsPlugin;
use crown_ash_client::plugins::network::CrownAshNetworkPlugin;
use crown_ash_client::plugins::map::CrownAshMapPlugin;
use crown_ash_client::plugins::ui::CrownAshUiPlugin;

fn main() {
    println!("Crown & Ash v0.3.5 starting...");
    println!("  Bevy renderer initializing (Vulkan/DX12/Metal)");
    println!("  27 GLB models embedded in binary");
    println!("  Controls: WASD = pan, scroll = zoom, click = select province");
    println!("  If you see only a dark window, try: WGPU_BACKEND=gl ./crown-ash-client");

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Crown & Ash".to_string(),
                resolution: (1600., 900.).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EmbeddedAssetsPlugin)
        .add_plugins(CrownAshNetworkPlugin)
        .add_plugins(CrownAshMapPlugin)
        .add_plugins(CrownAshUiPlugin)
        .run();
}
