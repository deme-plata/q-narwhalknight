/// Q-NarwhalKnight Beautiful Terminal UI
///
/// Provides an interactive, real-time dashboard for monitoring node status,
/// network metrics, blockchain state, and streaming logs.

pub mod app;
pub mod ui;
pub mod events;
pub mod metrics;

pub use app::App;
pub use events::{Event, EventHandler};

use anyhow::Result;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    Terminal,
};
use std::io;

/// Initialize the terminal for TUI mode
pub fn init_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;
    Ok(terminal)
}

/// Restore the terminal to normal mode
pub fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    Ok(())
}

/// Run the TUI application
pub async fn run_tui(mut app: App) -> Result<()> {
    let mut terminal = init_terminal()?;
    let mut event_handler = EventHandler::new(250); // 250ms tick rate

    loop {
        // Draw UI
        terminal.draw(|f| ui::render(f, &mut app))?;

        // Handle events
        match event_handler.next().await? {
            Event::Tick => {
                app.on_tick();
            }
            Event::Key(key) => {
                if app.handle_key_event(key) {
                    break; // User requested quit
                }
            }
            Event::Mouse(_) => {}
            Event::Resize(_, _) => {}
        }
    }

    restore_terminal(&mut terminal)?;
    Ok(())
}
