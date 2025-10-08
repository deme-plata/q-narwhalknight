pub mod dashboard;
pub mod cli;

#[cfg(feature = "gui")]
pub mod gui;

pub use dashboard::Dashboard;
pub use cli::CLIInterface;

#[cfg(feature = "gui")]
pub use gui::GuiApplication;
