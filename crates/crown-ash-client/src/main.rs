use bevy::prelude::*;
use crown_ash_client::plugins::network::CrownAshNetworkPlugin;
use crown_ash_client::plugins::map::CrownAshMapPlugin;
use crown_ash_client::plugins::ui::CrownAshUiPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Crown & Ash".to_string(),
                resolution: (1600., 900.).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(CrownAshNetworkPlugin)
        .add_plugins(CrownAshMapPlugin)
        .add_plugins(CrownAshUiPlugin)
        .run();
}
