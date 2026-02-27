pub mod dashboard;
pub mod cli;
pub mod tui_app;

#[cfg(feature = "tui")]
pub mod tui_views;

#[cfg(feature = "gui")]
pub mod gui;

pub use dashboard::Dashboard;
pub use cli::CLIInterface;
pub use tui_app::TuiApp;

#[cfg(feature = "gui")]
pub use gui::GuiApplication;

#[cfg(feature = "tui")]
pub use tui_app::run_tui;
