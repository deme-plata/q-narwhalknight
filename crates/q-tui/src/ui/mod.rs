pub mod bounty;
pub mod dashboard;
pub mod logs;
pub mod menu;
pub mod network;

use crate::app::{App, ViewMode};
use ratatui::Frame;

/// Main render function - dispatches to appropriate view
pub fn render(f: &mut Frame, app: &mut App) {
    match app.view_mode {
        ViewMode::Dashboard => dashboard::render(f, app),
        ViewMode::FullLogs => logs::render(f, app),
        ViewMode::Network => network::render(f, app),
        ViewMode::Menu => menu::render(f, app),
        ViewMode::Bounty => bounty::render(f, app),
    }
}
