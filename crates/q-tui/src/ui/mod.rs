pub mod bounty;
pub mod command_center;
pub mod dashboard;
pub mod logs;
pub mod menu;
pub mod network;
pub mod physics;
pub mod radar;
pub mod stats;
pub mod swarm_ocean;
pub mod wallet;
pub mod update_animation;
pub mod starship_animation;
pub mod water_robots_animation;
// q_animation.rs exists but is not compiled here — it's used directly by q-api-server's TUI mode

use crate::app::{App, ViewMode};
use ratatui::Frame;

/// Main render function - dispatches to appropriate view
pub fn render(f: &mut Frame, app: &mut App) {
    match app.view_mode {
        ViewMode::Dashboard => dashboard::render(f, app),
        ViewMode::Wallet => wallet::render(f, app),
        ViewMode::FullLogs => logs::render(f, app),
        ViewMode::Network => command_center::draw_command_center(f, f.size(), app),
        ViewMode::Stats => stats::render(f, app),
        ViewMode::Physics => physics::render(f, app),
        ViewMode::Menu => menu::render(f, app),
        ViewMode::Bounty => bounty::render(f, app),
    }

    // v9.8.5: Starship sync animation overlay (draws on top of sync views)
    if app.starship_animation.is_visible() {
        app.starship_animation.render(f.buffer_mut());
    }

    // v9.8.6: Water robots animation overlay (draws on important events)
    if app.water_robots.is_visible() {
        app.water_robots.render(f.buffer_mut());
    }

    // v9.8.4: Update animation overlay (draws on top of everything)
    if app.update_animation.is_visible() {
        app.update_animation.render(f.buffer_mut());
    }
}
