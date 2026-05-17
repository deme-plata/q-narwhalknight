#[cfg(feature = "tui")]
pub mod dashboard;
#[cfg(feature = "tui")]
pub mod diagnostics_view;
#[cfg(feature = "tui")]
pub mod network_view;
#[cfg(feature = "tui")]
pub mod events_view;
#[cfg(feature = "tui")]
pub mod settings_view;
#[cfg(feature = "tui")]
pub mod wallet_view;
#[cfg(feature = "tui")]
pub mod update_animation;
#[cfg(feature = "tui")]
pub mod q_animation;
#[cfg(feature = "tui")]
pub mod starship_animation;
#[cfg(feature = "tui")]
pub mod radar;
#[cfg(feature = "tui")]
pub mod swarm_ocean;
#[cfg(feature = "tui")]
pub mod command_center;

#[cfg(feature = "tui")]
use ratatui::Frame;
#[cfg(feature = "tui")]
use ratatui::layout::Rect;

#[cfg(feature = "tui")]
use super::tui_app::MinerTuiApp;

#[cfg(feature = "tui")]
pub fn draw_tab_content(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    match app.current_tab {
        0 => dashboard::draw_dashboard(f, area, app),
        1 => wallet_view::draw_wallet(f, area, app),
        2 => diagnostics_view::draw_diagnostics(f, area, app),
        3 => network_view::draw_network(f, area, app),
        4 => command_center::draw_command_center(f, area, app),
        5 => events_view::draw_events(f, area, app),
        6 => settings_view::draw_settings(f, area, app),
        _ => dashboard::draw_dashboard(f, area, app),
    }
}
